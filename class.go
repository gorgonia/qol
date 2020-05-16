package qol

import (
	"fmt"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

// Class represents the class ID of a dataset. It is an unbound type with a minimum of 0.
type Class uint

// ToClass converts a OneHotVector to a Class. This function panics if `a` is not a vector.
//
// The default threshold is 0.55 if 0 is passed in. The threshold does not apply to int tensors.
//
// Some behavioural notes:
// This function is NOT an argmax function. If the following vector is passed in,
// 	[0.1 0.1 0.6 0.7 0.1]
// the class returned will be 2, not 3.
//
// The same behaviour applies to `int` an `uint` tensor.
//
// For int Tensors, it assumes any value larger or equal to 1 is 1, and any value < 0 is 0. So the Class of the following:
//	[0 -1 3 3 1 0]
// will also be 2.
func ToClass(a tensor.Tensor, threshold float64) Class {
	if a.RequiresIterator() {
		panic(fmt.Sprintf("NYI: ToClass for a Tensor that requires an iterator. \n%v", a))
	}
	if !a.Shape().IsVector() {
		panic(fmt.Sprintf("ToClass only works on vectors. The shape of `a` was %v", a.Shape()))
	}

	if threshold == 0 {
		threshold = 0.55
	}

	data := a.Data()
	switch d := data.(type) {
	case []float32:
		thresh := float32(threshold)
		for i := range d {
			if d[i] > thresh {
				return Class(i)
			}
		}
	case []float64:
		thresh := threshold
		for i := range d {
			if d[i] > thresh {
				return Class(i)
			}
		}
	case []int:
		// no threshold
		for i := range d {
			if d[i] >= 1 {
				return Class(i)
			}
		}
	case []uint:
		// no threshold
		for i := range d {
			if d[i] >= 1 {
				return Class(i)
			}
		}
	default:
		panic(fmt.Sprintf("Data of type %T not implemented for ToClass()", data))
	}
	panic("Unreachable")
}

// ToClass converts a OneHotMatrix to Classes. This function panics if `a` is not a matrix.
// The default threshold is 0.55 if 0 is passed in. The threshold does not apply to int tensors.
//
// Some behavioural notes:
// This function is NOT an argmax function. If the following matrix is passed in,
// 	[0.1 0.1 0.6 0.7 0.1]
//	[0.6 0.1 0.1 0.7 0.1]
// the class returned will be [2 0], not [3 3].
//
// The same behaviour applies to `int` an `uint` tensor.
//
// For int Tensors, it assumes any value larger or equal to 1 is 1, and any value < 0 is 0. So the Class of the following:
//	[0 -1 3 3 1 0]
//	[2 0 -1 3 0 0]
// will also be [2 0] and not [2 3] or [3 3].
func ToClasses(a tensor.Tensor, threshold float64) []Class {
	if a.RequiresIterator() {
		panic(fmt.Sprintf("NYI: ToClass for a Tensor that requires an iterator. \n%v", a))
	}

	if !a.Shape().IsMatrix() {
		panic(fmt.Sprintf("ToClass only works on vectors. The shape of `a` was %v", a.Shape()))
	}

	if threshold == 0 {
		threshold = 0.55
	}

	iter, err := native.Matrix(a.(*tensor.Dense))
	if err != nil {
		panic(err)
	}
	retVal := make([]Class, a.Shape()[0])
	switch d := iter.(type) {
	case [][]float32:
		thresh := float32(threshold)
		for i := range d {
			for j := range d[i] {
				if d[i][j] > thresh {
					retVal[i] = Class(j)
					break
				}
				if j == len(d[i])-1 {
					panic(fmt.Sprintf("Unreachable class in matrix at row: %d", i))
				}
			}
		}
	case [][]float64:
		thresh := threshold
		for i := range d {
			for j := range d[i] {
				if d[i][j] > thresh {
					retVal[i] = Class(j)
					break
				}
				if j == len(d[i])-1 {
					panic(fmt.Sprintf("Unreachable class in matrix at row: %d", i))
				}
			}
		}
	case [][]int:
		for i := range d {
			for j := range d[i] {
				if d[i][j] >= 1 {
					retVal[i] = Class(j)
					break
				}
				if j == len(d[i])-1 {
					panic(fmt.Sprintf("Unreachable class in matrix at row: %d", i))
				}
			}
		}
	case [][]uint:
		for i := range d {
			for j := range d[i] {
				if d[i][j] >= 1 {
					retVal[i] = Class(j)
					break
				}
				if j == len(d[i])-1 {
					panic(fmt.Sprintf("Unreachable class in matrix at row: %d", i))
				}
			}
		}
	default:
		panic(fmt.Sprintf("Data of type %T not implemented for ToClasses()", iter))
	}
	return retVal
}

// ToOneHotVector converts a Class to a OneHotVector.
//
// The dtype defaults to tensor.Float64 if an empty Dtype was passed in.
func ToOneHotVector(a Class, numClasses uint, dtype tensor.Dtype) *tensor.Dense {
	if dtype.Type == nil {
		dtype = tensor.Float64
	}
	retVal := tensor.New(tensor.Of(dtype), tensor.WithShape(int(numClasses)))
	return UnsafeToOneHotVector(a, numClasses, retVal)
}

// ToOneHotMatrix converts a slice of Class to a OneHotMatrix.
//
// The dtype defaults to tensor.Float64 if an empty Dtype was passed in.
func ToOneHotMatrix(a []Class, numClasses uint, dtype tensor.Dtype) *tensor.Dense {
	if dtype.Type == nil {
		dtype = tensor.Float64
	}
	retVal := tensor.New(tensor.Of(dtype), tensor.WithShape(len(a), int(numClasses)))
	return UnsafeToOneHotMatrix(a, numClasses, retVal)
}

// UnsafeToOneHotVector converts a class to a OneHotVector, in the given
// tensor.Tensor. It expects the MaxClass the length of the given vector. Panics
// otherwise.
func UnsafeToOneHotVector(a Class, numClasses uint, reuse *tensor.Dense) *tensor.Dense {
	if !reuse.Shape().IsVector() {
		panic(fmt.Sprintf("UnsafeToOneHotVector only works on vectors. The shape of `reuse` was %v", reuse.Shape()))
	}
	if reuse.Shape()[0] != int(numClasses) {
		panic(fmt.Sprintf("UnsafeToOneHotVector expects length of `reuse`: %d to equal `numClasses`: %d", reuse.Shape()[0], int(numClasses)))
	}
	dt := reuse.Dtype()
	id := int(a)
	reuse.Zero()
	var err error
	switch dt {
	case tensor.Float32:
		err = reuse.SetAt(float32(1), id)
	case tensor.Float64:
		err = reuse.SetAt(float64(1), id)
	case tensor.Int64:
		err = reuse.SetAt(int64(1), id)
	case tensor.Int:
		err = reuse.SetAt(int(1), id)
	case tensor.Int32:
		err = reuse.SetAt(int32(1), id)
	default:
		panic(fmt.Sprintf("UnsafeToOneHotVector not implemented for %v", dt))
	}
	if err != nil {
		panic(err.Error())
	}
	return reuse
}

// UnsafeToOneHotMatrix converts a slice of Class to a OneHotMatrix, in the given tensor.Tensor. It expects a matrix of shape (len(a), numClasses). Panics otherwise.
func UnsafeToOneHotMatrix(a []Class, numClasses uint, reuse *tensor.Dense) *tensor.Dense {
	if !reuse.Shape().IsMatrix() {
		panic(fmt.Sprintf("UnsafeToOneHotMatrix only works on matracies. The shape of `reuse` was %v", reuse.Shape()))
	}
	if reuse.Shape()[0] != len(a) {
		panic(fmt.Sprintf("UnsafeToOneHotMatrix expects `len(a)` %d to be the number of rows in `reuse`: %d", len(a), reuse.Shape()[0]))
	}
	if reuse.Shape()[1] != int(numClasses) {
		panic(fmt.Sprintf("UnsafeToOneHotMatrix expects class dim (columns) of `reuse`: %d to equal `numClasses`: %d", reuse.Shape()[1], int(numClasses)))
	}
	dt := reuse.Dtype()
	reuse.Zero()
	var err error
	// Handline shapes of [1, numClasses]
	if reuse.IsRowVec() {
		switch dt {
		case tensor.Float32:
			reuse.Set(int(a[0]), float32(1))
		case tensor.Float64:
			reuse.Set(int(a[0]), float64(1))
		case tensor.Int64:
			reuse.Set(int(a[0]), int64(1))
		case tensor.Int:
			reuse.Set(int(a[0]), int(1))
		case tensor.Int32:
			reuse.Set(int(a[0]), int32(1))
		default:
			panic(fmt.Sprintf("UnsafeToOneHotVector not implemented for %v", dt))
		}
		return reuse
	}
	for i := range a {
		id := int(a[i]) //+
		switch dt {
		case tensor.Float32:
			err = reuse.SetAt(float32(1), i, id)
		case tensor.Float64:
			err = reuse.SetAt(float64(1), i, id)
		case tensor.Int64:
			err = reuse.SetAt(int64(1), i, id)
		case tensor.Int:
			err = reuse.SetAt(int(1), i, id)
		case tensor.Int32:
			err = reuse.SetAt(int32(1), i, id)
		default:
			panic(fmt.Sprintf("UnsafeToOneHotVector not implemented for %v", dt))
		}
		if err != nil {
			panic(err.Error())
		}
	}
	return reuse
}
