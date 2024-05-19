package pipeec

type NetworkConfig struct {
	WorldSize      int
	LocalRank      int
	FaultTolerance int
	Addrs          map[int]string
}
