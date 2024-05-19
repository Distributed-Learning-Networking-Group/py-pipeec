package pipeec

import (
	"log"
	"os"
	"syscall"
)

// MAPSIZE = 2GB
const MAPSIZE = 2 * 1024 * 1024 * 1024

type memorySegment struct {
	file *os.File
	data []byte
}

func mmap(file *os.File) *memorySegment {
	b, err := syscall.Mmap(
		int(file.Fd()),
		0,
		MAPSIZE,
		syscall.PROT_WRITE|syscall.PROT_READ,
		syscall.MAP_SHARED,
	)
	if err != nil {
		log.Fatal("mmap:", err)
	}

	return &memorySegment{
		file: file,
		data: b,
	}
}

func (s *memorySegment) munmap() {
	err := syscall.Munmap(s.data)
	if err != nil {
		log.Fatal(err)
	}
}
