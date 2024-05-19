package pipeec

import (
	"io"
	"log"
	"net"
	"os"
	"sync"
)

func handleIncomingConnection(conn net.Conn, wg *sync.WaitGroup) {
	defer conn.Close()
	defer wg.Done()
	var header MessageHeader
	err := header.Decode(conn)
	if err != nil {
		log.Default().Print("handle incomming connection error: ", err)
		return
	}

	if header.messageType == MESSAGE_WRITE {
		file, err := os.OpenFile(header.FileName, os.O_RDWR|os.O_CREATE, os.FileMode(0660))
		if err != nil {
			log.Default().Print("handle incomming connection error: ", err)
			return
		}
		defer file.Close()
		_, err = io.Copy(file, conn)
		if err != nil {
			log.Default().Print("handle incomming connection error: ", err)
			return
		}
	} else if header.messageType == MESSAGE_READ {
		file, err := os.Open(header.FileName)
		if err != nil {
			log.Default().Print("handle incomming connection error: ", err)
			return
		}
		defer file.Close()
		_, err = io.Copy(conn, file)
		if err != nil {
			log.Default().Print("handle incomming connection error: ", err)
			return
		}
	}

}

// for now, we don't provide shutdown primitives for simplicity
func StartListener(localAddr string) *sync.WaitGroup {
	listener, err := net.Listen("tcp", localAddr)
	if err != nil {
		log.Fatal(err)
	}

	wg := &sync.WaitGroup{}

	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				continue
			}
			wg.Add(1)
			handleIncomingConnection(conn, wg)
		}
	}()

	return wg
}
