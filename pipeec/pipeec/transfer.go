package pipeec

import (
	"fmt"
	"log"
	"net"
)

const TRANSFER_BLOCK_SIZE = 4 * 1024 * 1024 // 4MB

func transferTask(ctx *CheckPointContext, data []byte, destination int) {
	conn, err := net.Dial("tcp", ctx.config.Addrs[destination])
	var header MessageHeader
	if err != nil {
		goto fail_return
	}
	defer conn.Close()
	header.FileName = fmt.Sprintf("%s_%v", ctx.fileName, destination)
	header.messageType = MESSAGE_WRITE
	err = header.Encode(conn)
	if err != nil {
		goto fail_return
	}
	for len(data) > TRANSFER_BLOCK_SIZE {
		ctx.cond.L.Lock()
		if ctx.suspend {
			ctx.cond.Wait()
		}
		ctx.cond.L.Unlock()
		_, err = conn.Write(data[:TRANSFER_BLOCK_SIZE])
		if err != nil {
			goto fail_return
		}
		data = data[TRANSFER_BLOCK_SIZE:]
	}
	if len(data) > 0 {
		_, err := conn.Write(data)
		if err != nil {
			goto fail_return
		}
	}
	ctx.result <- 0 // failCount += 0
	return
fail_return:
	log.Default().Print("transfer task fail: ", err)
	ctx.result <- 1 // failCount += 1
}

func StartTransfer(ctx *CheckPointContext) {
	datas, err := ctx.enc.Split(ctx.Buffer())
	if err != nil {
		log.Fatal(err)
	}
	err = ctx.enc.Encode(datas)
	if err != nil {
		log.Fatal(err)
	}
	for destination := range datas {
		go transferTask(ctx, datas[destination], destination)
	}
}

func SuspendTransfer(ctx *CheckPointContext) {
	ctx.cond.L.Lock()
	ctx.suspend = true
	ctx.cond.L.Unlock()
}

func ResumeTransfer(ctx *CheckPointContext) {
	ctx.cond.L.Lock()
	ctx.suspend = false
	ctx.cond.L.Unlock()
	ctx.cond.Broadcast()
}

func WaitTransfer(ctx *CheckPointContext) {
	failCount := 0
	for range ctx.config.Addrs {
		failCount += <-ctx.result
	}

	if failCount > ctx.config.FaultTolerance {
		log.Fatalf("%v faults occured while only %v can be tolerated", failCount, ctx.config.FaultTolerance)
	}

}
