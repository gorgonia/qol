package qol

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// nodes creates new nodes. This will definitely change over time.

// OneHotVector creates a node that represents a one-hot vector.
func OneHotVector(a Class, numClasses uint, dtype tensor.Dtype, opts ...G.NodeConsOpt) *G.Node {
	ohv := ToOneHotVector(a, numClasses, dtype)
	return G.NewConstant(ohv, opts...)
}

// OneHotMatrix creates a node that represents a one-hot matrix.
func OneHotMatrix(a []Class, numClasses uint, dtype tensor.Dtype, opts ...G.NodeConsOpt) *G.Node {
	ohm := ToOneHotMatrix(a, numClasses, dtype)
	return G.NewConstant(ohm, opts...)
}
