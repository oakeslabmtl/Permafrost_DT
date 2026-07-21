"""Thaw front reconstruction service for 2D sensor inputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig, resolve_queue_config
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

SENSOR_QUEUE = "permafrost.record.sensors.2d"
THAW_FRONT_QUEUE = "permafrost.record.thaw_front.2d"
SENSOR_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"
THAW_FRONT_SCHEMA = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "thaw_front_message.json"


# =======================================================================================
# THAW FRONT RECONSTRUCTION SERVER
# =======================================================================================
class ThawFrontReconstructionServer:
    """Compute thaw front metrics and points from 2D sensor snapshots.

    Workflow:
    1. Receive sensor temperature message (CSV replay or real sensors)
    2. Parse sensor locations (x_m, z_m) and temperatures
    3. Interpolate temperature field onto a regular grid
    4. Extract 0°C isotherm as a smooth contour polyline
    5. Compute radius and volume metrics from the contour
    6. Publish results and store in InfluxDB
    """

    # =======================================================================================
    # INITIALIZATION AND CONFIGURATION
    # =======================================================================================
    def __init__(
        self,
        influx_config: InfluxConfig | None = None,
        input_config: RabbitMQConfig | None = None,
        output_config: RabbitMQConfig | None = None,
        *,
        input_queue: str | None = None,
        output_queue: str | None = None,
        thaw_threshold_c: float = 0.0,
        contour_grid_r: int = 120,
        contour_grid_z: int = 150,
        rmin_depth_bins: int = 8,
        surface_exclusion_m: float = 0.03,
        site_id: str = "default",
    ) -> None:
        """Initialize thaw front reconstruction service.

        Args:
            thaw_threshold_c: Temperature above which soil is considered thawed (default 0°C)
            contour_grid_r: Grid resolution in radial direction for interpolation
            contour_grid_z: Grid resolution in vertical direction for interpolation
            rmin_depth_bins: Number of depth slices used to compute radius_min_m
                (the least-thawed depth's outward radius); interior bins only.
            surface_exclusion_m: Depth band below the surface (z < this) whose
                contour points are ignored when computing the radius metrics. A
                near-surface horizontal 0°C layer (a real but safety-irrelevant
                experimental artefact) otherwise spans the full width and pins
                radius_max_m to the domain edge. The full contour is still
                published/plotted; only the radius metrics skip this band.
            site_id: Site identifier for InfluxDB storage

        Configuration flow:
          startup.conf → service_runners.py → this __init__ → configured instance ready to run
        """
        self.logger = get_logger("ThawFrontReconstructionServer")
        self.influx_config = influx_config or InfluxConfig()
        self.input_config = resolve_queue_config(
            input_config,
            queue=input_queue or SENSOR_QUEUE,
            schema_path=SENSOR_SCHEMA,
        )
        self.output_config = resolve_queue_config(
            output_config,
            queue=output_queue or THAW_FRONT_QUEUE,
            schema_path=THAW_FRONT_SCHEMA,
        )
        self.thaw_threshold_c = float(thaw_threshold_c)
        self.contour_grid_r = max(int(contour_grid_r), 10)
        self.contour_grid_z = max(int(contour_grid_z), 10)
        self.rmin_depth_bins = max(int(rmin_depth_bins), 3)
        self.surface_exclusion_m = max(float(surface_exclusion_m), 0.0)
        self.site_id = site_id

        self.mq_in: Optional[RabbitMQClient] = None
        self.mq_out: Optional[RabbitMQClient] = None
        self.influx: Optional[InfluxHelper] = None

    # =======================================================================================
    # DATA PARSING AND VALIDATION
    # =======================================================================================
    def _parse_sensor_samples(self, sensors: list[dict]) -> list[dict]:
        """Parse and validate raw sensor data from message.

        Input: Raw sensor list from sensor_message_2d.json
          [{"sensor_id": "D10-W15", "x_m": 0.1, "z_m": 0.15, "temperature": 0.25}, ...]

        Output: Standardized sensor samples (absolute x, float conversions)
          [{"x_m": 0.1, "z_m": 0.15, "temperature": 0.25}, ...]

        Handles missing/invalid data gracefully (skips, logs implicitly).
        """
        parsed: list[dict] = []
        for sensor in sensors:
            try:
                parsed.append(
                    {
                        "x_m": abs(float(sensor["x_m"])),  # <- Make x absolute (radial distance)
                        "z_m": float(sensor["z_m"]),
                        "temperature": float(sensor["temperature"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return parsed

    # =======================================================================================
    # THAW FRONT EXTRACTION (Interpolated)
    # =======================================================================================
    # Workflow:
    #   1. Build a regular (r, z) grid from the sensor bounding box
    #   2. Interpolate temperature field using scipy.griddata (linear + nearest fallback)
    #   3. Walk every grid cell to find edges where T crosses the threshold (marching squares)
    #   4. Connect crossing points into segments, trace into polylines
    #   5. Return the main (longest / widest) polyline as the thaw front
    # =======================================================================================

    @staticmethod
    def _iso_edge_point(
        p0: tuple[float, float],
        p1: tuple[float, float],
        v0: float,
        v1: float,
        threshold: float,
    ) -> tuple[float, float] | None:
        """Find isotherm crossing point on an edge via linear interpolation.

        Returns the (x, z) point where the value equals threshold, or None if
        the edge does not cross the threshold.
        """
        if not np.isfinite(v0) or not np.isfinite(v1):
            return None
        d0 = v0 - threshold
        d1 = v1 - threshold
        if d0 == 0.0 and d1 == 0.0:
            return None
        if d0 == 0.0:
            return p0
        if d1 == 0.0:
            return p1
        if d0 * d1 > 0.0:
            return None
        denom = v1 - v0
        if denom == 0.0:
            return None
        ratio = (threshold - v0) / denom
        return float(p0[0] + ratio * (p1[0] - p0[0])), float(p0[1] + ratio * (p1[1] - p0[1]))

    @staticmethod
    def _point_key(point: tuple[float, float], ndigits: int = 10) -> tuple[float, float]:
        """Round coordinates for deduplication across floating-point precision boundaries."""
        return round(point[0], ndigits), round(point[1], ndigits)

    def _extract_main_isocontour(
        self,
        T_grid: np.ndarray,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        threshold_c: float,
    ) -> list[dict]:
        """Extract the 0°C isotherm from the interpolated temperature grid.

        Walks every grid cell, finds edges where T crosses threshold_c, builds
        connected segments, traces them into polylines, and returns the main one
        (longest, with largest radial extent).

        Output: [{"x_m": ..., "z_m": ...}, ...] ordered along the contour.
        """
        nz, nr = T_grid.shape
        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        for j in range(nz - 1):
            z0, z1 = float(z_grid[j]), float(z_grid[j + 1])
            for i in range(nr - 1):
                r0, r1 = float(r_grid[i]), float(r_grid[i + 1])

                v00 = T_grid[j, i]
                v10 = T_grid[j, i + 1]
                v11 = T_grid[j + 1, i + 1]
                v01 = T_grid[j + 1, i]
                if not (np.isfinite(v00) and np.isfinite(v10) and np.isfinite(v11) and np.isfinite(v01)):
                    continue

                p00, p10, p11, p01 = (r0, z0), (r1, z0), (r1, z1), (r0, z1)

                edge_points: dict[int, tuple[float, float]] = {}
                for key, args in (
                    (0, (p00, p10, float(v00), float(v10))),
                    (1, (p10, p11, float(v10), float(v11))),
                    (2, (p11, p01, float(v11), float(v01))),
                    (3, (p01, p00, float(v01), float(v00))),
                ):
                    pt = self._iso_edge_point(*args, threshold_c)
                    if pt is not None:
                        edge_points[key] = pt

                if len(edge_points) < 2:
                    continue
                if len(edge_points) == 2:
                    ids = list(edge_points.keys())
                    p_a, p_b = edge_points[ids[0]], edge_points[ids[1]]
                    if p_a != p_b:
                        segments.append((p_a, p_b))
                elif len(edge_points) == 4:
                    # Ambiguous saddle: resolve using cell-centre value
                    center = float(v00 + v10 + v11 + v01) / 4.0
                    pairings = [(0, 1), (2, 3)] if center >= threshold_c else [(0, 3), (1, 2)]
                    for e0, e1 in pairings:
                        p_a, p_b = edge_points[e0], edge_points[e1]
                        if p_a != p_b:
                            segments.append((p_a, p_b))
                else:
                    ids = list(edge_points.keys())
                    for idx in range(len(ids) - 1):
                        p_a, p_b = edge_points[ids[idx]], edge_points[ids[idx + 1]]
                        if p_a != p_b:
                            segments.append((p_a, p_b))

        if not segments:
            return []

        # Build adjacency graph
        neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
        coords: dict[tuple[float, float], tuple[float, float]] = {}
        for p_a, p_b in segments:
            k0, k1 = self._point_key(p_a), self._point_key(p_b)
            if k0 == k1:
                continue
            coords[k0] = p_a
            coords[k1] = p_b
            neighbors.setdefault(k0, set()).add(k1)
            neighbors.setdefault(k1, set()).add(k0)

        if not neighbors:
            return []

        def walk_component(start: tuple[float, float], comp: set[tuple[float, float]]) -> list[tuple[float, float]]:
            degree1 = [k for k in comp if len(neighbors.get(k, set())) == 1]
            current = degree1[0] if degree1 else start
            prev: tuple[float, float] | None = None
            ordered = [current]
            visited_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
            while True:
                next_key = None
                for cand in neighbors.get(current, set()):
                    if cand == prev:
                        continue
                    edge = (current, cand) if current < cand else (cand, current)
                    if edge not in visited_edges:
                        next_key = cand
                        visited_edges.add(edge)
                        break
                if next_key is None:
                    break
                ordered.append(next_key)
                prev, current = current, next_key
                if not degree1 and current == ordered[0]:
                    break
            return [coords[k] for k in ordered]

        # Find connected components and trace each into a polyline
        components: list[set[tuple[float, float]]] = []
        seen: set[tuple[float, float]] = set()
        for key in neighbors:
            if key in seen:
                continue
            stack = [key]
            comp: set[tuple[float, float]] = set()
            while stack:
                k = stack.pop()
                if k in comp:
                    continue
                comp.add(k)
                for n in neighbors.get(k, set()):
                    if n not in comp:
                        stack.append(n)
            seen |= comp
            components.append(comp)

        polylines = [walk_component(next(iter(c)), c) for c in components]
        polylines = [p for p in polylines if len(p) > 1]
        if not polylines:
            return []

        main_poly = max(polylines, key=lambda poly: (len(poly), max(abs(p[0]) for p in poly)))
        return [{"x_m": float(p[0]), "z_m": float(p[1])} for p in main_poly]

    def _reconstruct_thaw_front(self, samples: list[dict]) -> list[dict]:
        """Reconstruct the smooth 0°C isotherm from sensor measurements.

        1. Interpolate sensor temperatures onto a regular (r, z) grid
        2. Fill edge NaNs with nearest-neighbour
        3. Extract the isotherm via marching squares

        Returns a list of ordered (x_m, z_m) points forming the thaw boundary,
        or an empty list if the contour cannot be determined.
        """
        if len(samples) < 4:
            return []

        r_vals = np.array([float(s["x_m"]) for s in samples], dtype=float)
        z_vals = np.array([float(s["z_m"]) for s in samples], dtype=float)
        temp_vals = np.array([float(s["temperature"]) for s in samples], dtype=float)
        if not np.any(np.isfinite(temp_vals)):
            return []

        r_grid = np.linspace(float(np.min(r_vals)), float(np.max(r_vals)), self.contour_grid_r)
        z_grid = np.linspace(float(np.min(z_vals)), float(np.max(z_vals)), self.contour_grid_z)
        R, Z = np.meshgrid(r_grid, z_grid)
        pts = np.column_stack((r_vals, z_vals))

        T_grid = griddata(pts, temp_vals, (R, Z), method="linear")
        if np.all(np.isnan(T_grid)):
            return []

        if np.isnan(T_grid).any():
            T_grid = np.where(np.isnan(T_grid), griddata(pts, temp_vals, (R, Z), method="nearest"), T_grid)

        return self._extract_main_isocontour(T_grid, r_grid, z_grid, self.thaw_threshold_c)

    # =======================================================================================
    # METRIC COMPUTATION
    # =======================================================================================
    def _compute_radius_metrics(self, points: list[dict]) -> tuple[float | None, float | None, float | None]:
        """Compute max, mean and min radial extent of the thaw front.

        radius_max_m / radius_avg_m are the largest / mean radial distance of the
        isotherm points from the pipe axis.

        radius_min_m is the *least-thawed* outward radius: the thaw front is binned
        by depth, the outward radius R(z)=max|x_m| is taken per slice, and the
        smallest such radius over interior slices is returned. The naive min over
        all contour points is useless here because the isotherm curls back to the
        axis (r->0) at its top/bottom extremes, pinning it at 0. This depth-wise
        definition instead tracks the weakest heating direction and grows with
        time, making it usable as an actuation gate (heat while r_min < limit).

        A shallow near-surface band (z < surface_exclusion_m) is dropped first:
        the physical experiment shows a horizontal 0°C layer just below the
        surface that spans the full radial width, which would otherwise pin
        radius_max_m to the domain edge and mask the true radial thaw front
        driven by the heater. If too few points remain to define a front, we
        fall back to the full set rather than returning nothing.

        Output: (radius_max_m, radius_avg_m, radius_min_m)
        """
        if not points:
            return None, None, None
        metric_points = [
            p for p in points
            if float(p["z_m"]) >= self.surface_exclusion_m
        ]
        if len(metric_points) < 3:
            # Not enough below-surface structure — use everything to stay robust.
            metric_points = points
        radii = np.array([abs(float(p["x_m"])) for p in metric_points], dtype=float)
        z_vals = np.array([float(p["z_m"]) for p in metric_points], dtype=float)
        radius_max_m = float(radii.max())
        radius_avg_m = float(radii.mean())

        z0, z1 = float(z_vals.min()), float(z_vals.max())
        if z1 <= z0:
            # Degenerate (single depth): no depth structure to exploit.
            return radius_max_m, radius_avg_m, radius_max_m

        nbins = self.rmin_depth_bins
        edges = np.linspace(z0, z1, nbins + 1)
        bin_idx = np.clip(np.digitize(z_vals, edges) - 1, 0, nbins - 1)
        outer_per_bin = [float(radii[bin_idx == b].max()) for b in range(nbins) if np.any(bin_idx == b)]
        # Drop the top/bottom bins where the contour clips the domain edge and
        # collapses to the axis; fall back to all bins if too few remain.
        interior = outer_per_bin[1:-1] if len(outer_per_bin) > 2 else outer_per_bin
        radius_min_m = float(min(interior))
        return radius_max_m, radius_avg_m, radius_min_m

    def _compute_volume_metric(self, points: list[dict]) -> float | None:
        """Compute thaw volume by integrating disc cross-sections along depth (frustum method).

        Assumes axial symmetry around the heating pipe. At each depth z, the thaw
        cross-section is a disc of radius R(z) = x_m. Between consecutive depth
        slices the boundary is linear, so each slice is a frustum:
            V_i = (π·h_i / 3) · (R_i² + R_i·R_{i+1} + R_{i+1}²)

        Points are sorted by z_m before integration to ensure correct stacking.

        Input:  List of points [{"x_m": ..., "z_m": ...}, ...]
        Output: Total thaw volume in m³, or None if fewer than 2 points.
        """
        if len(points) < 2:
            return None
        sorted_pts = sorted(points, key=lambda p: float(p["z_m"]))
        volume = 0.0
        for i in range(len(sorted_pts) - 1):
            r0 = abs(float(sorted_pts[i]["x_m"]))
            r1 = abs(float(sorted_pts[i + 1]["x_m"]))
            h = abs(float(sorted_pts[i + 1]["z_m"]) - float(sorted_pts[i]["z_m"]))
            volume += (np.pi * h / 3.0) * (r0 * r0 + r0 * r1 + r1 * r1)
        return volume

    # =======================================================================================
    # MESSAGE PROCESSING AND OUTPUT
    # =======================================================================================
    def _process_message(self, msg: dict) -> None:
        """Process a single sensor snapshot and publish thaw-front metrics."""
        if self.influx is None or self.mq_out is None:
            raise RuntimeError("Dependencies not initialised. Did you call setup()?")  # pragma: no cover

        time_hours = float(msg["time_hours"])
        samples = self._parse_sensor_samples(msg.get("sensors", []))
        points = self._reconstruct_thaw_front(samples)
        radius_max_m, radius_avg_m, radius_min_m = self._compute_radius_metrics(points)
        thaw_volume_m3 = self._compute_volume_metric(points)

        payload = {
            "timestamp": msg.get("timestamp") or datetime.utcnow().isoformat(),
            "time_hours": time_hours,
            "radius_max_m": radius_max_m,
            "radius_avg_m": radius_avg_m,
            "radius_min_m": radius_min_m,
            "thaw_volume_m3": thaw_volume_m3,
            "points": points,
            "point_count": len(points),
        }

        self.mq_out.publish(payload)
        self.influx.write_thaw_front_metrics(
            time_hours=time_hours,
            radius_max_m=radius_max_m,
            radius_avg_m=radius_avg_m,
            radius_min_m=radius_min_m,
            thaw_volume_m3=thaw_volume_m3,
            points=points,
            point_count=len(points),
            site=self.site_id,
        )
        self.logger.info(
            "Thaw front updated (t=%.2fh, contour_points=%d)",
            time_hours,
            len(points),
        )

    # =======================================================================================
    # SERVICE LIFECYCLE
    # =======================================================================================
    def setup(self) -> None:
        """Initialize RabbitMQ and InfluxDB connections.

        Called before start() to establish dependencies.
        Separated from __init__ to support lazy connection (only when running).
        """
        if self.mq_in is None:
            self.mq_in = RabbitMQClient(self.input_config)
        if self.mq_out is None:
            self.mq_out = RabbitMQClient(self.output_config)
        if self.influx is None:
            self.influx = InfluxHelper(self.influx_config)
        self.logger.info(
            "Dependencies initialised (in=%s, out=%s)",
            self.input_config.queue,
            self.output_config.queue,
        )

    def start(self) -> None:
        """Start the service: connect to RabbitMQ and listen for sensor messages.

        Workflow:
        1. Call setup() to initialize connections
        2. Start consuming messages from sensor queue
        3. For each message, call _process_message()
        4. Catch interrupt (Ctrl+C) and shut down gracefully
        """
        if self.mq_in is None or self.mq_out is None or self.influx is None:
            self.setup()

        self.logger.info("Listening for sensor updates on %s", self.input_config.queue)
        try:
            self.mq_in.consume(callback=self._process_message)
        except KeyboardInterrupt:  # pragma: no cover - runtime behaviour
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        """Stop listening for messages (non-blocking)."""
        if self.mq_in and self.mq_in.channel:
            self.mq_in.channel.stop_consuming()

    def close(self) -> None:
        """Disconnect all services and clean up resources."""
        self.stop()
        if self.mq_in is not None:
            self.mq_in.disconnect()
        if self.mq_out is not None:
            self.mq_out.disconnect()
        if self.influx is not None:
            self.influx.close()
        self.logger.info("Shutdown complete")


# =======================================================================================
# ENTRY POINT
# =======================================================================================
if __name__ == "__main__":
    ThawFrontReconstructionServer().start()
