import sys
import os

sys.argv = [
    sys.argv[0],
    '--codec', 'pcm',
    '--aec', 'webrtc',
    '--vad', 'none',
    '--denoiser', 'webrtc_mild',
    '--arch', 'm2'
]

import teacher_node
if __name__ == "__main__":
    teacher_node.main()
