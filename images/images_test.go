package imqol

import (
	"fmt"
	"os"
)

func ExampleJPEG() {
	filename := "testdata/smile.jpeg"
	f, err := os.Open(filename)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	t, err := JPEG(f)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	fmt.Printf("%v %v", t.Shape(), t.Dtype)

	// Output:
	// (12, 14, 4) float64
}

func ExamplePNG() {
	filename := "testdata/smile.png"
	f, err := os.Open(filename)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	t, err := PNG(f)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	fmt.Printf("%v %v", t.Shape(), t.Dtype())

	// Output:
	// (12, 14, 4) float64
}
