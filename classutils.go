package qol

import (
	"sort"

	"github.com/xtgo/set"
)

type Classes []Class

func (cs Classes) Len() int           { return len(cs) }
func (cs Classes) Less(i, j int) bool { return cs[i] < cs[j] }
func (cs Classes) Swap(i, j int)      { cs[i], cs[j] = cs[j], cs[i] }

// Distinct returns a Classes with only the distinct classes. This method clobbers the original order of data.
func (cs Classes) Distinct() Classes {
	sort.Sort(cs)
	n := set.Uniq(cs)
	return cs[:n]
}

func (cs Classes) Clone() Classes { retVal := make(Classes, len(cs)); copy(retVal, cs); return retVal }
