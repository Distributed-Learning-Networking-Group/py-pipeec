package pipeec_test

import (
	"bytes"
	"fmt"
	"os"
	"pipeec/pipeec"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func generateAddrs(basePort uint16, worldSize int) map[int]string {
	ret := make(map[int]string)
	for i := 0; i < worldSize; i += 1 {
		ret[i] = fmt.Sprintf("localhost:%v", basePort+uint16(i))
	}
	return ret
}

func generateConfig(worldSize int, faultTolerance int, addrBasePort uint16) []*pipeec.NetworkConfig {
	addrs := generateAddrs(addrBasePort, worldSize)
	ret := make([]*pipeec.NetworkConfig, worldSize)
	for i := 0; i < worldSize; i += 1 {
		ret[i] = &pipeec.NetworkConfig{
			WorldSize:      worldSize,
			LocalRank:      i,
			FaultTolerance: faultTolerance,
			Addrs:          addrs,
		}
	}
	return ret
}

var contents []byte = bytes.Repeat([]byte("the answer is 42."), 114)

func tempFile() (*os.File, error) {
	f, err := os.CreateTemp(".", "pipeec_test_file*")
	if err != nil {
		return nil, err
	}
	f.Write(contents)
	return f, nil
}

func remove(filename string, part int) {
	os.Remove(filename)
	for i := 0; i < part; i += 1 {
		os.Remove(fmt.Sprintf("%s_%v", filename, i))
	}
}

func TestTransfer(t *testing.T) {
	configs := generateConfig(2, 1, 43391)
	ctxs := make([]*pipeec.CheckPointContext, 2)
	wgs := make([]*sync.WaitGroup, 2)

	file, err := tempFile()
	if err != nil {
		t.Fatal(err)
	}
	defer remove(file.Name(), 2)

	for i := range configs {
		ctxs[i] = pipeec.InitCheckPointContext(file, configs[i])
		wgs[i] = pipeec.StartListener(configs[i].Addrs[configs[i].LocalRank])
	}

	ctx := ctxs[0]

	pipeec.StartTransfer(ctx)
	pipeec.WaitTransfer(ctx)

	wgs[0].Wait()
	wgs[1].Wait()

	read_buffer := make([]byte, len(contents))

	pipeec.Read(ctx, read_buffer)

	assert.True(t, bytes.Equal(read_buffer, contents))

	pipeec.DestoryCheckPointContext(ctxs[0])

}
