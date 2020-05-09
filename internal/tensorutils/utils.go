package tensorutils

import (
	"fmt"

	"gorgonia.org/tensor"
)

func GetDense(x tensor.Tensor) *tensor.Dense {
	switch t := x.(type) {
	case *tensor.Dense:
		return t
	case tensor.Densor:
		return t.Dense()
	default:
		panic(fmt.Sprintf("Cannot retrieve *Dense from x of %T", x))
	}
}
