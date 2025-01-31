# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from subprocess import check_output
from tempfile import NamedTemporaryFile
from typing import Sequence, Tuple, Union

import numpy as np
import trimesh
import trimesh.scene
from cached_property import cached_property
from shapely import ops
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.ops import snap, triangulate
from trimesh.exchange import gltf

from .utils.math import rotate_around_point
from .lanepoints import LanePoints

from smarts.core.utils.sumo import sumolib  # isort:skip
from sumolib.net.edge import Edge  # isort:skip
from sumolib.net.lane import Lane  # isort:skip


def _convert_camera(camera):
    result = {
        "name": camera.name,
        "type": "perspective",
        "perspective": {
            "aspectRatio": camera.fov[0] / camera.fov[1],
            "yfov": np.radians(camera.fov[1]),
            "znear": float(camera.z_near),
            # HACK: The trimesh gltf export doesn't include a zfar which Panda3D GLB
            #       loader expects. Here we override to make loading possible.
            "zfar": float(camera.z_near + 100),
        },
    }
    return result


gltf._convert_camera = _convert_camera


@dataclass
class LaneData:
    sumo_lane: Lane
    lane_speed: float


@dataclass
class RoadEdgeData:
    forward_edges: Sequence[Edge]
    oncoming_edges: Sequence[Edge]


class GLBData:
    def __init__(self, bytes_):
        self._bytes = bytes_

    def write_glb(self, output_path):
        with open(output_path, "wb") as f:
            f.write(self._bytes)


class SumoRoadNetwork:
    # 3.2 is the default Sumo road network lane width if it's not specified
    # explicitly in Sumo's NetEdit or the map.net.xml file.
    # This corresponds on a 1:1 scale to lanes 3.2m wide, which is typical
    # in North America (although US highway lanes are wider at ~3.7m).
    DEFAULT_LANE_WIDTH = 3.2

    def __init__(
        self, graph, net_file, default_lane_width=None, lanepoint_spacing=None
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._graph = graph
        self._net_file = net_file
        self._default_lane_width = (
            default_lane_width
            if default_lane_width is not None
            else SumoRoadNetwork.DEFAULT_LANE_WIDTH
        )
        self._lanepoints = None
        if lanepoint_spacing is not None:
            assert lanepoint_spacing > 0
            # XXX: this should be last here since LanePoints() calls road_network methods immediately
            self._lanepoints = LanePoints(self, spacing=lanepoint_spacing)

    @staticmethod
    def _check_net_origin(bbox):
        assert len(bbox) == 4
        return bbox[0] <= 0.0 and bbox[1] <= 0.0 and bbox[2] >= 0.0 and bbox[3] >= 0.0

    shifted_net_file_name = "shifted_map-AUTOGEN.net.xml"

    @classmethod
    def shifted_net_file_path(cls, net_file_path):
        net_file_folder = os.path.dirname(net_file_path)
        return os.path.join(net_file_folder, cls.shifted_net_file_name)

    @classmethod
    @lru_cache(maxsize=1)
    def _shift_coordinates(cls, net_file_path, shifted_path):
        assert shifted_path != net_file_path
        logger = logging.getLogger(cls.__name__)
        logger.info(f"normalizing net coordinates into {shifted_path}...")
        ## Translate the map's origin to remove huge (imprecise) offsets.
        ## See https://sumo.dlr.de/docs/netconvert.html#usage_description
        ## for netconvert options description.
        try:
            stdout = check_output(
                [
                    "netconvert",
                    "--offset.disable-normalization=FALSE",
                    "-s",
                    net_file_path,
                    "-o",
                    shifted_path,
                ]
            )
            logger.debug(f"netconvert output: {stdout}")
            return True
        except Exception as e:
            logger.warning(
                f"unable to use netconvert tool to normalize coordinates: {e}"
            )
        return False

    @classmethod
    def from_file(
        cls,
        net_file,
        shift_to_origin=False,
        default_lane_width=None,
        lanepoint_spacing=None,
    ):
        # Connections to internal lanes are implicit. If `withInternal=True` is
        # set internal junctions and the connections from internal lanes are
        # loaded into the network graph.
        G = sumolib.net.readNet(net_file, withInternal=True)

        if not cls._check_net_origin(G.getBoundary()):
            shifted_net_file = cls.shifted_net_file_path(net_file)
            if os.path.isfile(shifted_net_file) or (
                shift_to_origin and cls._shift_coordinates(net_file, shifted_net_file)
            ):
                G = sumolib.net.readNet(shifted_net_file, withInternal=True)
                assert cls._check_net_origin(G.getBoundary())
                net_file = shifted_net_file
                # keep track of having shifted the graph by
                # injecting state into the network graph.
                # this is needed because some maps have been pre-shifted,
                # and will already have a locationOffset, but for those
                # the offset should not be used (because all their other
                # coordinates are relative to the origin).
                G._shifted_by_smarts = True

        return cls(
            G,
            net_file,
            default_lane_width=default_lane_width,
            lanepoint_spacing=lanepoint_spacing,
        )

    @property
    def graph(self):
        return self._graph

    @property
    def net_file(self):
        """This is the net file (*.net.xml) that corresponds with our possibly-offset coordinates."""
        return self._net_file

    @cached_property
    def net_offset(self):
        """This is our offset from what's in the original net file."""
        return (
            self.graph.getLocationOffset()
            if self.graph and getattr(self.graph, "_shifted_by_smarts", False)
            else [0, 0]
        )

    @property
    def default_lane_width(self):
        return self._default_lane_width

    @default_lane_width.setter
    def default_lane_width(self, default_lane_width):
        self._default_lane_width = default_lane_width

    @property
    def lanepoints(self):
        return self._lanepoints

    def _compute_road_polygons(self):
        lane_to_poly = {}
        for edge in self._graph.getEdges():
            for lane in edge.getLanes():
                shape = SumoRoadNetwork._buffered_lane_or_edge(lane, lane.getWidth())
                # Check if "shape" is just a point.
                if len(set(shape.exterior.coords)) == 1:
                    logging.debug(
                        f"Lane:{lane.getID()} has provided non-shape values {lane.getShape()}"
                    )
                    continue

                lane_to_poly[lane.getID()] = shape

        # Remove holes created at tight junctions due to crude map geometry
        self._snap_internal_holes(lane_to_poly)
        self._snap_external_holes(lane_to_poly)
        # Remove break in visible lane connections created when lane enters an intersection
        self._snap_internal_edges(lane_to_poly)

        polys = list(lane_to_poly.values())

        for node in self._graph.getNodes():
            line = node.getShape()
            if len(line) <= 2 or len(set(line)) == 1:
                self._log.debug(
                    "Skipping {}-type node with <= 2 vertices".format(node.getType())
                )
                continue

            polys.append(Polygon(line))

        return polys

    def _snap_internal_edges(self, lane_to_poly, snap_threshold=2):
        # HACK: Internal edges that have tight curves, when buffered their ends do not
        #       create a tight seam with the connected lanes. This procedure attempts
        #       to remedy that with snapping.
        for lane_id in lane_to_poly:
            lane = self.lane_by_id(lane_id)

            # Only do snapping for internal edge lanes
            if not lane.getEdge().isSpecial():
                continue

            lane_shape = lane_to_poly[lane_id]
            incoming = self.lane_by_id(lane_id).getIncoming()[0]
            incoming_shape = lane_to_poly.get(incoming.getID())
            if incoming_shape:
                lane_shape = Polygon(snap(lane_shape, incoming_shape, snap_threshold))
                lane_to_poly[lane_id] = lane_shape

            outgoing = self.lane_by_id(lane_id).getOutgoing()[0].getToLane()
            outgoing_shape = lane_to_poly.get(outgoing.getID())
            if outgoing_shape:
                lane_shape = Polygon(snap(lane_shape, outgoing_shape, snap_threshold))
                lane_to_poly[lane_id] = lane_shape

    def _snap_internal_holes(self, lane_to_poly, snap_threshold=2):
        for lane_id in lane_to_poly:
            lane = self.lane_by_id(lane_id)

            # Only do snapping for internal edge lane holes
            if not lane.getEdge().isSpecial():
                continue
            lane_shape = lane_to_poly[lane_id]
            for x, y in lane_shape.exterior.coords:
                for nl, dist in self.nearest_lanes(
                    (x, y),
                    max(10, 2 * self._default_lane_width),
                    include_junctions=False,
                ):
                    if not nl:
                        continue
                    nl_shape = lane_to_poly.get(nl.getID())
                    if nl_shape:
                        lane_shape = Polygon(snap(lane_shape, nl_shape, snap_threshold))
            lane_to_poly[lane_id] = lane_shape

    def _snap_external_holes(self, lane_to_poly, snap_threshold=2):
        for lane_id in lane_to_poly:
            lane = self.lane_by_id(lane_id)

            # Only do snapping for external edge lane holes
            if lane.getEdge().isSpecial():
                continue

            incoming = self.lane_by_id(lane_id).getIncoming()
            if incoming and incoming[0].getEdge().isSpecial():
                continue

            outgoing = self.lane_by_id(lane_id).getOutgoing()
            if outgoing:
                outgoing_lane = outgoing[0].getToLane()
                if outgoing_lane.getEdge().isSpecial():
                    continue

            lane_shape = lane_to_poly[lane_id]
            for x, y in lane_shape.exterior.coords:
                for nl, dist in self.nearest_lanes(
                    (x, y),
                    max(10, 2 * self._default_lane_width),
                    include_junctions=False,
                ):
                    if (not nl) or (nl and nl.getEdge().isSpecial()):
                        continue
                    nl_shape = lane_to_poly.get(nl.getID())
                    if nl_shape:
                        lane_shape = Polygon(snap(lane_shape, nl_shape, snap_threshold))
            lane_to_poly[lane_id] = lane_shape

    @staticmethod
    def _triangulate(polygon):
        return [
            tri_face
            for tri_face in triangulate(polygon)
            if tri_face.centroid.within(polygon)
        ]

    def _make_glb_from_polys(self, polygons):
        scene = trimesh.Scene()
        vertices, faces = [], []
        point_dict = dict()
        current_point_index = 0

        # Trimesh's API require a list of vertices and a list of faces, where each
        # face contains three indexes into the vertices list. Ideally, the vertices
        # are all unique and the faces list references the same indexes as needed.
        # TODO: Batch the polygon processing.
        for poly in polygons:
            # Collect all the points on the shape to reduce checks by 3 times
            for x, y in poly.exterior.coords:
                p = (x, y, 0)
                if p not in point_dict:
                    vertices.append(p)
                    point_dict[p] = current_point_index
                    current_point_index += 1
            triangles = SumoRoadNetwork._triangulate(poly)
            for triangle in triangles:
                face = np.array(
                    [point_dict.get((x, y, 0), -1) for x, y in triangle.exterior.coords]
                )
                # Add face if not invalid
                if -1 not in face:
                    faces.append(face)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Trimesh doesn't support a coordinate-system="z-up" configuration, so we
        # have to apply the transformation manually.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
        )

        # Attach additional information for rendering as metadata in the map glb
        metadata = {}

        # <2D-BOUNDING_BOX>: four floats separated by ',' (<FLOAT>,<FLOAT>,<FLOAT>,<FLOAT>),
        # which describe x-minimum, y-minimum, x-maximum, and y-maximum
        metadata["bounding_box"] = self.graph.getBoundary()

        # lane markings information
        lane_dividers, edge_dividers = self.compute_traffic_dividers()
        metadata["lane_dividers"] = lane_dividers
        metadata["edge_dividers"] = edge_dividers

        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial()
        )

        scene.add_geometry(mesh)
        return GLBData(gltf.export_glb(scene, extras=metadata, include_normals=True))

    def build_glb(self) -> GLBData:
        polys = self._compute_road_polygons()
        return self._make_glb_from_polys(polys)

    def edge_by_id(self, edge_id):
        return self._graph.getEdge(edge_id)

    def lane_by_id(self, lane_id):
        return self._graph.getLane(lane_id)

    def lane_by_index_on_edge(self, edge_id, lane_idx):
        return self.edge_by_id(edge_id).getLane(lane_idx)

    def edge_by_lane_id(self, lane_id):
        return self.lane_by_id(lane_id).getEdge()

    def road_edge_data_for_lane_id(self, lane_id: str) -> RoadEdgeData:
        lane_edge = self.edge_by_lane_id(lane_id)
        from_node, to_node = lane_edge.getFromNode(), lane_edge.getToNode()
        edges_going_the_same_way = [
            edge
            for edge in from_node.getOutgoing()
            if edge.getToNode().getID() == to_node.getID()
        ]
        edges_going_the_opposite_way = [
            edge
            for edge in to_node.getOutgoing()
            if edge.getToNode().getID() == from_node.getID()
        ]

        return RoadEdgeData(
            forward_edges=edges_going_the_same_way,
            oncoming_edges=edges_going_the_opposite_way,
        )

    def lane_vector_at_offset(self, lane: Lane, start_offset: float) -> np.ndarray:
        """Computes the lane direction vector at the given offset into the road."""

        add_offset = 1
        end_offset = start_offset + add_offset  # a little further down the lane
        lane_length = lane.getLength()

        if end_offset > lane_length + add_offset:
            raise ValueError(
                f"Offset={end_offset} goes out further than the end of "
                f"lane=({lane.getID()}, length={lane_length})"
            )

        p1 = self.world_coord_from_offset(lane, start_offset)
        p2 = self.world_coord_from_offset(lane, end_offset)

        return p2 - p1

    @classmethod
    def buffered_lane_or_edge(cls, lane_or_edge: Union[Edge, Lane], width: float = 1.0):
        return list(cls._buffered_lane_or_edge(lane_or_edge, width).exterior.coords)

    @staticmethod
    def _buffered_lane_or_edge(lane_or_edge: Union[Edge, Lane], width: float = 1.0):
        shape = lane_or_edge.getShape()
        ls = LineString(shape).buffer(
            width / 2,
            1,
            cap_style=CAP_STYLE.flat,
            join_style=JOIN_STYLE.round,
            mitre_limit=5.0,
        )

        if isinstance(ls, Polygon):
            buffered_shape = ls
        elif isinstance(ls, MultiPolygon):
            # Sometimes it oddly outputs a MultiPolygon and then we need to turn it into a convex
            #  hull
            buffered_shape = ls.convex_hull
        else:
            raise RuntimeError("Shapely `object.buffer` behavior may have changed.")

        return buffered_shape

    def lane_data_for_lane(self, lane):
        lane_speed = lane.getSpeed()
        return LaneData(sumo_lane=lane, lane_speed=lane_speed)

    def nearest_lanes(
        self, point, radius=None, include_junctions=True, include_special=True
    ) -> Sequence[Tuple[Lane, float]]:
        """The closest lanes to the given point

        Args:
            point: (x, y) find the nearest lanes to this coordinate
            radius: The max distance to search around point for a lane
            include_junctions: Whether the shape of a junction should be considered when computing distance to lane
            include_special: Whether to include lanes who are on "special" edges (for example 'internal' edges)
        """

        if radius is None:
            radius = max(10, 2 * self._default_lane_width)
        x, y = point
        # XXX: note that this getNeighboringLanes() call is fairly heavy/expensive (as revealed by profiling)
        candidate_lanes = self._graph.getNeighboringLanes(
            x, y, r=radius, includeJunctions=include_junctions, allowFallback=False
        )

        if not include_special:
            candidate_lanes = [
                lane for lane in candidate_lanes if not lane[0].getEdge().isSpecial()
            ]

        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])

        return candidate_lanes

    def nearest_lane(
        self, point, radius=None, include_junctions=True, include_special=True
    ) -> Lane:
        """Find the nearest lane to the given point

        Args:
            point: (x, y) find the nearest lane to this coordinate
            radius: The max distance to search around point for a lane
            include_junctions: Whether the shape of a junction should be considered when computing distance to lane
            include_special: Whether to include lanes who are on "special" edges (for example 'internal' edges)
        """
        if radius is None:
            radius = max(10, 2 * self._default_lane_width)
        nearest_lanes = self.nearest_lanes(
            point, radius, include_junctions, include_special
        )
        return nearest_lanes[0][0] if nearest_lanes else None

    def lane_center_at_point(self, lane: Lane, point) -> np.ndarray:
        lane_offset = self.offset_into_lane(lane, point)
        lane_center_at_offset = self.world_coord_from_offset(lane, lane_offset)
        return lane_center_at_offset

    def world_to_lane_coord(self, lane: Lane, point) -> np.ndarray:
        """Maps a world coordinate to a lane local coordinate

        Args:
            lane: sumo lane object
            point: (x, y) world space coordinate

        Returns:
            np.array([u, v]) where
                u is meters into the lane,
                v is signed distance from center (right of center is negative)
        """

        u = self.offset_into_lane(lane, point)
        lane_vector = self.lane_vector_at_offset(lane, u)
        lane_normal = np.array([-lane_vector[1], lane_vector[0]])

        lane_center_at_u = self.world_coord_from_offset(lane, u)
        offcenter_vector = np.array(point) - lane_center_at_u

        v_sign = np.sign(np.dot(offcenter_vector, lane_normal))
        v = np.linalg.norm(offcenter_vector) * v_sign

        return np.array([u, v])

    def offset_into_lane(self, lane, point):
        """Calculate how far (in meters) into the lane the given point is.

        Args:
            point: (x, y) find the position on the lane to this coordinate
            lane: sumo network lane

        Returns:
            offset_into_lane: meters from start of lane
        """
        lane_shape = lane.getShape(False)

        offset_into_lane = 0
        # SUMO geomhelper.polygonOffset asserts when the point is part of the shape.
        # We get around the assertion with a check if the point is part of the shape.
        if tuple(point) in lane_shape:
            for i in range(len(lane_shape) - 1):
                if lane_shape[i] == point:
                    break
                offset_into_lane += sumolib.geomhelper.distance(
                    lane_shape[i], lane_shape[i + 1]
                )
        else:
            offset_into_lane = (
                sumolib.geomhelper.polygonOffsetWithMinimumDistanceToPoint(
                    point, lane_shape, perpendicular=False
                )
            )

        return offset_into_lane

    def split_lane_shape_at_offset(
        self, lane_shape: Polygon, lane: Lane, offset: float
    ):
        width_2 = lane.getWidth()
        point = self.world_coord_from_offset(lane, offset)
        lane_vec = self.lane_vector_at_offset(lane, offset)

        perp_vec_right = rotate_around_point(lane_vec, np.pi / 2, origin=(0, 0))
        perp_vec_right = (
            perp_vec_right / max(np.linalg.norm(perp_vec_right), 1e-3) * width_2 + point
        )

        perp_vec_left = rotate_around_point(lane_vec, -np.pi / 2, origin=(0, 0))
        perp_vec_left = (
            perp_vec_left / max(np.linalg.norm(perp_vec_left), 1e-3) * width_2 + point
        )

        split_line = LineString([perp_vec_left, perp_vec_right])
        return ops.split(lane_shape, split_line)

    def world_coord_from_offset(self, lane: Lane, offset) -> np.ndarray:
        """Convert offset into lane to world coordinates

        Args:
            offset: meters into the lane
            lane: sumo network lane

        Returns:
            np.array([x, y]): coordinates in world space
        """
        lane_shape = lane.getShape(False)

        position_on_lane = sumolib.geomhelper.positionAtShapeOffset(lane_shape, offset)

        return np.array(position_on_lane)

    def point_is_within_road(self, point):
        # XXX: Not robust around junctions (since their shape is quite crude?)
        radius = max(5, 2 * self._default_lane_width)
        for nl, dist in self.nearest_lanes(point[:2], radius=radius):
            if dist < 0.5 * nl.getWidth() + 1e-1:
                return True
        return False

    def drove_off_map(
        self, veh_position: Tuple[float, float, float], veh_heading: float
    ) -> bool:
        # try to determine if the vehicle "exited" the map by driving beyond the end of a dead-end lane.
        radius = max(5, 2 * self._default_lane_width)
        nearest_lanes = self.nearest_lanes(veh_position[:2], radius=radius)
        if not nearest_lanes:
            return False  # we can't tell anything here
        nl, dist = nearest_lanes[0]
        if nl.getOutgoing() or dist < 0.5 * nl.getWidth() + 1e-1:
            return False  # the last lane it was in was not a dead-end, or it's still in a lane
        end_node = nl.getEdge().getToNode()
        end_point = end_node.getCoord()
        dist = math.sqrt(
            (veh_position[0] - end_point[0]) ** 2
            + (veh_position[1] - end_point[1]) ** 2
        )
        if dist > 2 * nl.getWidth():
            return False  # it's no where near the end of the lane
        # now check its heading to ensure it was going in roughly the right direction for this lane
        end_shape = end_node.getShape()
        veh_heading %= 2 * math.pi
        tolerance = math.pi / 4
        for p in range(1, len(end_shape)):
            num = end_shape[p][1] - end_shape[p - 1][1]
            den = end_shape[p][0] - end_shape[p - 1][0]
            crossing_heading = math.atan(-den / num)
            if den < 0:
                crossing_heading += math.pi
            elif num < 0:
                crossing_heading -= math.pi
            crossing_heading -= math.pi / 2
            # we allow for it to be going either way since it's a pain to determine which side of the edge it's on
            if (
                abs(veh_heading - crossing_heading % (2 * math.pi)) < tolerance
                or abs(
                    (veh_heading + math.pi) % (2 * math.pi)
                    - crossing_heading % (2 * math.pi)
                )
                < tolerance
            ):
                return True
        return False

    def road_nodes_with_triggers(self):
        """Scan the road network for any nodes with ID's that match the form:

            .*=trigger[<spawn_node_id_1>@<spawn_node_id_2>@...]

        A node with an ID with this pattern signals to the simulator that
        when an agent comes within some radius of this node, we should spawn
        social vehicles at the listed spawn nodes

        Returns:
            [(trigger_node, [spawn_nodes])]: list of all trigger nodes and their
                                            spawn nodes in the network.
        """
        nodes = self._graph.getNodes()
        nodes_with_triggers = []
        for node in nodes:
            # example node id with trigger:
            #  'gneJ2=trigger[source1@source2@source3]'
            matches = re.match(r".*=trigger\[(.*)\]", node.getID())
            if matches is None:
                continue

            spawn_node_ids = matches.group(1).split("@")

            for spawn_node_id in spawn_node_ids:
                # verify that these spawn nodes exist
                assert self._graph.hasNode(spawn_node_id)

            spawn_nodes = [self._graph.getNode(s) for s in spawn_node_ids]
            nodes_with_triggers.append((node, spawn_nodes))

        return nodes_with_triggers

    def random_route_starting_at_edge(self, edge, max_route_len=10):
        route = []
        curr_edge = edge
        while len(route) < max_route_len:
            route.append(curr_edge.getID())

            next_edges = list(curr_edge.getOutgoing().keys())
            if not next_edges:
                break

            curr_edge = random.choice(next_edges)

        return route

    def random_route_starting_at_node(self, node, max_route_len=10):
        route = []

        if not node.getOutgoing():
            # this node is terminating
            return route

        edge = random.choice(node.getOutgoing())
        return self.random_route_starting_at_edge(edge, max_route_len)

    def get_edge_in_junction(
        self, start_edge_id, start_lane_index, end_edge_id, end_lane_index
    ):
        start_edge = self._graph.getEdge(start_edge_id)
        start_lane = start_edge.getLane(start_lane_index)
        end_edge = self._graph.getEdge(end_edge_id)
        end_lane = end_edge.getLane(end_lane_index)
        connection = start_lane.getConnection(end_lane)

        # If there is no connection beween try and do the best
        if connection is None:
            # The first id is good enough since we just need to determine the junction edge id
            connection = start_edge.getConnections(end_edge)[0]

        connection_lane_id = connection.getViaLaneID()
        connection_lane = self._graph.getLane(connection_lane_id)

        return connection_lane.getEdge().getID()

    def random_route(self, max_route_len=10):
        edge = random.choice(self._graph.getEdges(False))
        return self.random_route_starting_at_edge(edge, max_route_len)

    def compute_traffic_dividers(self, threshold=1):
        lane_dividers = []  # divider between lanes with same traffic direction
        edge_dividers = []  # divider between lanes with opposite traffic direction
        edge_borders = []
        for edge in self._graph.getEdges():
            # Omit intersection for now
            if edge.getFunction() == "internal":
                continue

            lanes = edge.getLanes()
            for i in range(len(lanes)):
                shape = lanes[i].getShape()
                left_side = sumolib.geomhelper.move2side(
                    shape, -lanes[i].getWidth() / 2
                )
                right_side = sumolib.geomhelper.move2side(
                    shape, lanes[i].getWidth() / 2
                )

                if i == 0:
                    edge_borders.append(right_side)

                if i == len(lanes) - 1:
                    edge_borders.append(left_side)
                else:
                    lane_dividers.append(left_side)

        # The edge borders that overlapped in positions form an edge divider
        for i in range(len(edge_borders) - 1):
            for j in range(i + 1, len(edge_borders)):
                edge_border_i = np.array(
                    [edge_borders[i][0], edge_borders[i][-1]]
                )  # start and end position
                edge_border_j = np.array(
                    [edge_borders[j][-1], edge_borders[j][0]]
                )  # start and end position with reverse traffic direction

                # The edge borders of two lanes do not always overlap perfectly, thus relax the tolerance threshold to 1
                if np.linalg.norm(edge_border_i - edge_border_j) < threshold:
                    edge_dividers.append(edge_borders[i])

        return lane_dividers, edge_dividers
