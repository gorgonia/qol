package gchack

var ballast []byte

func init() {
	ballast = make([]byte, 2<<30)
}
