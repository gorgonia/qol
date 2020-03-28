package imqol

import (
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type options struct {
	as    tensor.Dtype
	reuse *tensor.Dense
}

// FnOpt is an option to configure the running of a function.
type FnOpt interface {
	do(*options)
}

// AsType is a construction option
type AsType int

const (
	AsFloat64 AsType = iota
	AsFloat32
)

func (a AsType) do(opt *options) {
	switch a {
	case AsFloat64:
		opt.as = tensor.Float64
	case AsFloat32:
		opt.as = tensor.Float32
	default:
		panic(fmt.Sprintf("Type %v unsupported as a construction option", a))
	}
}

// JPEG loads a JPEG file into a *tensor.Dense of HWC format.
func JPEG(f io.Reader, opts ...FnOpt) (*tensor.Dense, error) {
	return decoder(f, jpeg.Decode, opts...)
}

// PNG loads a PNG file into a *tensor.Dense of HWC format
func PNG(f io.Reader, opts ...FnOpt) (*tensor.Dense, error) {
	return decoder(f, png.Decode, opts...)
}

type decoderFn func(r io.Reader) (image.Image, error)

func decoder(f io.Reader, fn decoderFn, opts ...FnOpt) (*tensor.Dense, error) {
	im, err := fn(f)
	if err != nil {
		return nil, err
	}

	// HANDLE THE DECODING OF SUBSAMPLED IMAGES
	// Subsampled images do not play well with deep learning applications.
	// So we discretize them to RGBA
	//
	// To do this, we draw a new RGBA and use that as the "image"
	switch im.(type) {
	case *image.YCbCr:
		im2 := image.NewRGBA(image.Rect(0, 0, im.Bounds().Dx(), im.Bounds().Dy()))
		draw.Draw(im2, im2.Bounds(), im, im.Bounds().Min, draw.Src)
		im = im2
	}

	opt := options{
		as: tensor.Float64,
	}
	for _, o := range opts {
		o.do(&opt)
	}

	height := im.Bounds().Dy()
	width := im.Bounds().Dx()
	chans := channels(im)
	px := pix(im)

	v, err := rawToFloats(px, opt.as)
	if err != nil {
		return nil, err
	}

	retVal := tensor.New(tensor.WithShape(height, width, chans), tensor.WithBacking(v))
	return retVal, nil

}

func channels(im image.Image) int {
	switch im.(type) {
	case *image.RGBA:
		return 4
	case *image.NRGBA:
		return 4
	case *image.Gray:
		return 1
	case *image.Gray16:
		return 1
	default:
		panic(fmt.Sprintf("Unhandled image.Image type %T", im))
	}
	panic("Unreachable")
}

func pix(im image.Image) []byte {
	switch a := im.(type) {
	case *image.RGBA:
		return a.Pix
	case *image.NRGBA:
		return a.Pix
	case *image.Gray:
		return a.Pix
	case *image.Gray16:
		return a.Pix
	default:
		panic(fmt.Sprintf("Unhandled image.Image type %T", im))
	}
	panic("Unreachable")
}

func rawToFloats(a []byte, as tensor.Dtype) (interface{}, error) {
	switch as {
	case tensor.Float64:
		return bytesToFloat64(a), nil
	case tensor.Float32:
		return bytesToFloat32(a), nil
	}
	return nil, errors.Errorf("Dtype %v not supported in converting to float", as)
}

func bytesToFloat64(a []byte) []float64 {
	retVal := make([]float64, len(a))
	for i, n := range a {
		retVal[i] = float64(n)
	}
	return retVal
}

func bytesToFloat32(a []byte) []float32 {
	retVal := make([]float32, len(a))
	for i, n := range a {
		retVal[i] = float32(n)
	}
	return retVal
}
