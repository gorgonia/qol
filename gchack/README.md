# gchack #

package memhacks is a good way to hack around Go's garbage collection. While the hack is easy to write for each program, it would be quite tedious to manually manage them.

The trick is to create a ballast as described here: https://blog.twitch.tv/go-memory-ballast-how-i-learnt-to-stop-worrying-and-love-the-heap-26c2462549a2

# Usage #

Simply import the library like so:

```
import _ gorgonia.org/qol/gchack
```
