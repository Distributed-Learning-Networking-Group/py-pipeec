package main

import (
	"C"
)
import (
	"bytes"
	"encoding/json"
	"log"
	"os"
	"pipeec/pipeec"
)

var svcs []*checkPointService

func service(svcId serviceId) *checkPointService {
	return svcs[svcId]
}

//export PipeecInitService
func PipeecInitService(localAddrC *C.char) int {
	localAddr := C.GoString(localAddrC)
	svc := &checkPointService{
		wg: pipeec.StartListener(localAddr),
	}
	id := len(svcs)
	svcs = append(svcs, svc)
	return id
}

//export PipeecInitCheckPointContext
func PipeecInitCheckPointContext(svcId int, fileNameC *C.char, networkConfigJsonC *C.char) int {
	networkConfig := C.GoString(networkConfigJsonC)
	fileName := C.GoString(fileNameC)
	config := &pipeec.NetworkConfig{}
	dec := json.NewDecoder(bytes.NewBuffer([]byte(networkConfig)))
	err := dec.Decode(&config)
	if err != nil {
		return -1
	}
	file, err := os.OpenFile(fileName, os.O_RDWR, 0)
	if err != nil {
		return -1
	}
	ctx := pipeec.InitCheckPointContext(file, config)
	return int(service(serviceId(svcId)).InsertContext(ctx))
}

//export PipeecDestroyCheckPointContext
func PipeecDestroyCheckPointContext(svcId int, ctxId int) {
	svc := service(serviceId(svcId))
	ctx := svc.GetContext(contextId(ctxId))
	if ctx != nil {
		pipeec.DestoryCheckPointContext(ctx)
		svc.RemoveContext(contextId(ctxId))
	}
}

//export PipeecStartTransfer
func PipeecStartTransfer(svcId int, ctxId int) {
	ctx := service(serviceId(svcId)).GetContext(contextId(ctxId))
	if ctx == nil {
		log.Fatal("Pipeec Start Tansfer: invalid context id: ", ctxId)
	}
	pipeec.StartTransfer(ctx)
}

//export PipeecSuspendTransfer
func PipeecSuspendTransfer(svcId int, ctxId int) {
	ctx := service(serviceId(svcId)).GetContext(contextId(ctxId))
	if ctx == nil {
		log.Fatal("Pipeec Start Tansfer: invalid context id: ", ctxId)
	}
	pipeec.SuspendTransfer(ctx)
}

//export PipeecResumeTransfer
func PipeecResumeTransfer(svcId int, ctxId int) {
	ctx := service(serviceId(svcId)).GetContext(contextId(ctxId))
	if ctx == nil {
		log.Fatal("Pipeec Start Tansfer: invalid context id: ", ctxId)
	}
	pipeec.ResumeTransfer(ctx)
}

//export PipeecRead
func PipeecRead(svcId int, ctxId int) {
	ctx := service(serviceId(svcId)).GetContext(contextId(ctxId))
	if ctx == nil {
		log.Fatal("Pipeec Start Tansfer: invalid context id: ", ctxId)
	}
	pipeec.Read(ctx, ctx.Buffer())
}

//export PipeecWaitTransfer
func PipeecWaitTransfer(svcId int, ctxId int) {
	ctx := service(serviceId(svcId)).GetContext(contextId(ctxId))
	if ctx == nil {
		log.Fatal("Pipeec Start Tansfer: invalid context id: ", ctxId)
	}
	pipeec.WaitTransfer(ctx)
}

func main() {

}
