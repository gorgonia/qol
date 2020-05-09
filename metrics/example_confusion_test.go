package metrics_test

import (
	"fmt"

	"gonum.org/v1/plot/vg"
	"gorgonia.org/qol"
	. "gorgonia.org/qol/metrics"
)

func Example_Confusion() {
	pred := qol.Classes{0, 2, 2, 2, 4, 1, 1, 1}
	correct := qol.Classes{0, 0, 2, 1, 3, 2, 1, 1}

	mat := Confusion(pred, correct)
	fmt.Printf("%v", mat)

	p, err := mat.Heatmap()
	if err != nil {
		fmt.Printf("Failed to create a heatmap plot. %v", err)
	}
	if err := p.Save(4*vg.Centimeter, 4*vg.Centimeter, "testdata/heatmap.png"); err != nil {
		fmt.Printf("Failed to save heatmap plot. %v.\n", err)
	}

	// Output:
	// ⎡1  0  0  0  0⎤
	// ⎢0  2  1  0  0⎥
	// ⎢1  1  1  0  0⎥
	// ⎢0  0  0  0  0⎥
	// ⎣0  0  0  1  0⎦

}
