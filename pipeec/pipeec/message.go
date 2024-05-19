package pipeec

import (
	"encoding/binary"
	"io"
)

const (
	MESSAGE_WRITE = iota
	MESSAGE_READ
)

type MessageHeader struct {
	FileName    string
	messageType byte
	Length      int64 //reserved
	Timestamp   int64 //reserved
}

func (msg *MessageHeader) Encode(writer io.Writer) error {
	length := int64(len(msg.FileName))
	err := binary.Write(writer, binary.LittleEndian, msg.messageType)
	if err != nil {
		return err
	}
	err = binary.Write(writer, binary.LittleEndian, length)
	if err != nil {
		return err
	}
	_, err = writer.Write([]byte(msg.FileName))
	if err != nil {
		return err
	}
	return nil
}

func (msg *MessageHeader) Decode(reader io.Reader) error {
	var length int64
	err := binary.Read(reader, binary.LittleEndian, &msg.messageType)
	if err != nil {
		return err
	}
	err = binary.Read(reader, binary.LittleEndian, &length)
	if err != nil {
		return err
	}
	fileNameBuffer := make([]byte, length)
	_, err = reader.Read(fileNameBuffer)
	if err != nil {
		return err
	}
	msg.FileName = string(fileNameBuffer)
	return nil
}
