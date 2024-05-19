package main

import (
	"pipeec/pipeec"
	"sync"
)

type contextId int
type serviceId int

type checkPointService struct {
	ctxs []*pipeec.CheckPointContext
	wg   *sync.WaitGroup
}

func (svc *checkPointService) InsertContext(ctx *pipeec.CheckPointContext) contextId {
	free := 0
	length := len(svc.ctxs)
	for i := range svc.ctxs {
		if svc.ctxs[i] == nil {
			free = i
			break
		}
	}
	if free == length {
		svc.ctxs = append(svc.ctxs, ctx)
	} else {
		svc.ctxs[free] = ctx
	}
	return contextId(free)
}

func (svc *checkPointService) RemoveContext(id contextId) {
	svc.ctxs[id] = nil
}

func (svc *checkPointService) GetContext(id contextId) *pipeec.CheckPointContext {
	return svc.ctxs[id]
}
