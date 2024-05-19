package pipeec

import (
	"log"
	"os"
	"sync"

	"github.com/klauspost/reedsolomon"
)

type CheckPointContext struct {
	fileName string
	enc      reedsolomon.Encoder
	segment  *memorySegment
	config   *NetworkConfig
	suspend  bool
	cond     *sync.Cond
	result   chan int
}

func (ctx *CheckPointContext) Buffer() []byte {
	fileStat, err := ctx.segment.file.Stat()
	if err != nil {
		log.Fatal("ctx.Buffer:", err)
	}
	return ctx.segment.data[:fileStat.Size()]
}

func InitCheckPointContext(file *os.File, config *NetworkConfig) *CheckPointContext {
	segment := mmap(file)
	dataShards := config.WorldSize - config.FaultTolerance
	parityShards := config.FaultTolerance
	enc, err := reedsolomon.New(dataShards, parityShards)
	if err != nil {
		log.Fatal(err)
	}
	return &CheckPointContext{
		fileName: file.Name(),
		enc:      enc,
		segment:  segment,
		config:   config,
		suspend:  false,
		cond:     sync.NewCond(&sync.Mutex{}),
		result:   make(chan int),
	}
}

func DestoryCheckPointContext(ctx *CheckPointContext) {
	ctx.segment.munmap()
	ctx.segment.file.Close()
}
