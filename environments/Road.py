import traci

class Road:
    def __init__(self):
        """
        assume all lanes have the same width
        """
        # self.warmupEdge = 'warm_up'
        self.highwayEntEdge = 'highway_in'
        # self.entranceEdgeID = 'entranceEdge'
        self.rampEntEdgeID = 'ramp_ent'
        self.highwayOutEdgeID = 'highway_out'

        self.highwayKeepRouteID = 'keep_on_highway'
        self.rampMergingRouteID = 'ramp_merge'

        self.rampEntEdgeLaneID_0 = self.rampEntEdgeID  + '_0'
        self.highwayOutEdgeLaneID_0=self.highwayOutEdgeID+'_0'
        self.speedLimitRamp = traci.lane.getMaxSpeed(self.rampEntEdgeLaneID_0)
        self.speedLimitHwout=traci.lane.getMaxSpeed(self.highwayOutEdgeLaneID_0)
        #
        # self.laneNum = traci.edge.getLaneNumber(self.entranceEdgeID)
        # self.laneWidth = traci.lane.getWidth(self.entranceEdgeLaneID_0)
        # self.laneLength = traci.lane.getLength(self.entranceEdgeLaneID_0)
        self.mergingPoint= traci.junction.getPosition('highway_out_j')
        self.destJunction = traci.junction.getPosition('destination')
        self.highwayinj = traci.junction.getPosition('highway_in_j')
        self.rampstartj=traci.junction.getPosition('ramp_start_j')