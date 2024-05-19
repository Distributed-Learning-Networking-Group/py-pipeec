package pipeec

import (
	"fmt"
	"io"
	"log"
	"net"
)

func readTask(ctx *CheckPointContext, results [][]byte, destination int) {
	conn, err := net.Dial("tcp", ctx.config.Addrs[destination])
	header := MessageHeader{
		FileName:    fmt.Sprintf("%s_%v", ctx.fileName, destination),
		messageType: MESSAGE_READ,
	}
	var content []byte

	if err != nil {
		goto fail_return
	}
	err = header.Encode(conn)
	if err != nil {
		goto fail_return
	}

	content, err = io.ReadAll(conn)
	if err != nil {
		goto fail_return
	}
	results[destination] = content
	ctx.result <- 0
	return
fail_return:
	log.Default().Print("transfer task fail: ", err)
	ctx.result <- 1
}

func copy_chunks(chunks [][]byte, destination []byte) {
	for _, chunk := range chunks {
		nbytes := copy(destination, chunk)
		destination = destination[nbytes:]
	}
}

func make_consistent(chunks [][]byte) {
	maxLen := 0
	// find the max length
	for i := range chunks {
		if chunks[i] == nil {
			continue
		}
		maxLen = max(maxLen, len(chunks[i]))
	}

	// if the filesize is not maxLen, the file
	// is truncated

	for i := range chunks {
		if chunks[i] == nil {
			continue
		}

		if len(chunks[i]) != maxLen {
			chunks[i] = nil
		}
	}
}

func Read(ctx *CheckPointContext, buffer []byte) {
	worldSize := ctx.config.WorldSize
	results := make([][]byte, worldSize)
	for i := 0; i < worldSize; i += 1 {
		go readTask(ctx, results, i)
	}

	failCount := 0

	for range results {
		failCount += <-ctx.result
	}

	make_consistent(results)

	err := ctx.enc.ReconstructData(results)
	if err != nil {
		log.Fatal("Read: ", err)
	}
	ok, err := ctx.enc.Verify(results)
	if !ok || err != nil {
		log.Fatal("Read: Verify fail, ", err)
	}
	if failCount > ctx.config.FaultTolerance {
		log.Fatalf("%v faults occured while only %v can be tolerated", failCount, ctx.config.FaultTolerance)
	}

	copy_chunks(results[:worldSize-ctx.config.FaultTolerance], buffer)

}
