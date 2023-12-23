#!/usr/bin/env python3

"""
Sound converter for Ultima 8.

Ultima 8 Sound Format:
file   = header {frame}.       # Frames readed until end of file
header = size rate ver {byte}  # 32 byte header padded with zero bytes
         [segtab].             # segment table appear for big sounds (>32767)
size   = dword.                # Size of output buffer
rate   = word.                 # Samples per second
ver    = byte.                 # Equals 1

segtab = segmark seg seg       #
         segend {byte}.        # padded with zero
segmark= dword.                # = 32
seg    = offset bufsize        # offset to frame, full buffer size + 32
segend = dword.                # offset to end of file

frame  = fsize fusize chksum   #
         mode nfact {factors}  #
         {opcode | zbits}.     # zbits used for modes [15..22]
fsize  = word.                 # Size of frame (including fsize itself)
fusize = word.                 # Decompressed size of frame
chksum = word.                 # Checksum
mode   = byte.                 # [0..7]    -> ?
                               # [8..14]   -> MinData in [6..0]
                               # [15..22]  -> MinData in [6..0] + zero bit
                               # [23..255] -> ?
nfact  = byte.                 # Number of factors for LPC encoding
factors= {short}.              # N facctors for LPC (data)

zbits  = "0" | "1" opcode.     # 0 -> Decompress zero value
opcode = op0 | op1 | op2.      #
op0    = "0" {bit}.            # Read MinData+1 bit value (two-complement)
op1    = {"1"} "0" {bit}.      # Count ones to get Size. Max Size = 7-MinData-2.
                               # Than read MinData+Size bit value.
op2    = {"1"} {bit}.          # Read 7-MinData-1 ones. Than read 7 bit value.

Note: op1 and op2 data encoded with some Excess-like Code.
      See unpack_excess() and pack_excess() for more information.
Note: Bytes in stream are in big-endian form.
"""

import os
import sys
import struct

from typing import IO, Union, TYPE_CHECKING
from array import ArrayType

__author__ = "DeaDDooMER"
__license__ = "MIT"
__version__ = "1.0.0"

if sys.version_info < (3, 9):
    raise Exception("Python 3.9+ required")

if TYPE_CHECKING:
    ArrayOfInt = ArrayType[int]
else:
    ArrayOfInt = ArrayType

class BitStream:
    data: bytearray
    offset: int

    def __init__(self, data: bytearray, offset: int = 0) -> None:
        assert(offset >= 0)
        assert(offset <= len(data) * 8)
        self.data = data
        self.offset = offset

    def read_bit(self) -> Union[bool, int]:
        if self.offset + 1 > len(self.data) * 8:
            raise ValueError("failed to read 1 bit: end of stream")
        bit: bool = (self.data[self.offset // 8] >> (self.offset % 8)) & 1 != 0
        self.offset += 1
        return bit

    def write_bit(self, value: Union[bool, int]) -> None:
        if self.offset >= len(self.data) * 8:
            self.data.append(0)
        self.data[self.offset // 8] &= ~(1 << (self.offset % 8))
        self.data[self.offset // 8] |= (value != 0) << (self.offset % 8)
        self.offset += 1

    def read_int(self, bits: int) -> int:
        v: int = 0
        i: int = 0
        if self.offset + bits > len(self.data) * 8:
            raise ValueError(f"failed to read {bits:d} bits: end of stream")
        for i in range(bits):
            v |= self.read_bit() << i
        return v

    def write_int(self, value: int, bits: int) -> None:
        i: int
        value &= (1 << bits) - 1
        for i in range(bits):
            self.write_bit((value >> i) & 1)


def sign_extend(value: int, bits: int) -> int:
    # Convert two's complement value to python integer
    assert(value >= 0)
    assert(bits >= 0)
    sign_bit: int = 1 << bits - 1
    return (value & sign_bit - 1) - (value & sign_bit)

def in_two_complement(value: int, bits: int) -> bool:
    limit: int = 1 << bits - 1
    return value >= -limit and value < limit

def unpack_excess(value: int, bits: int) -> int:
    # Convert some Excess-like Code to python integer.
    # This representation have two integer ranges for negative and positive
    # values, which depends on bit-length of value.
    # There is a table which contents representable values:
    #
    # Bits |    Negative Value    |   Positive Value   | Comment
    #   0  |        0             |           0        |  0 not representable!
    #   1  |       -2             |           1        | -1 not representable!
    #   2  |    [-4..-3]          |        [2..3]      |
    #   3  |    [-8..-5]          |        [4..7]      |
    #   4  |   [-16..-9]          |        [8..15]     |
    #   5  |   [-32..-17]         |       [16..31]     |
    #   6  |   [-64..-33]         |       [32..63]     |
    #   7  |  [-128..-65]         |       [64..127]    |
    #   N  | [-2**N..-2**(N-1)-1] | [2**(N-1)..2**N-1] |
    #
    assert(value >= 0)
    assert(bits >= 0)
    sign_bit: int = 1 << bits
    return (value & sign_bit - 1) - (~value << 1 & sign_bit)

def pack_excess(value: int) -> tuple[int, int]:
    # Pack python integer to Special Excess Code (see unpack_excess())
    # Return: (unsigned value, bits needed for this value)
    # Note: 0 and -1 are not representable in this format
    #       in such cases returted (value=0, bits=0)
    bits: int = (value + (value < 0)).bit_length()
    sign_bit: int = 1 << bits
    return (value - sign_bit) & (sign_bit - 1), bits

def in_excess(value: int, bits: int) -> bool:
    # Check that `value` representable in Special Excess Code of `bits` bits
    a: int = 1 << bits
    b: int = 1 << bits - 1
    return value >= -a and value < -b or value >= b and value < a

# ============================================================================ #
#                        Ultima 8 Sound Format (Reader)                        #
# ============================================================================ #

def decode_bitstream(mode: int, n: int, src: bytearray) -> bytearray:
    s: BitStream       # input buffer
    dst: bytearray     # output buffer
    zero: bool = False # stream includes special bit
    sample: int = 0    # decoded sample byte (signed)
    size: int = 0      # decoded opcode (size)
    i: int = 0         # bit counter
    j: int = 0         # current output sample
    e: int = 0         # (debug) expected opcode size
    offset: int = 0    # (debug) opcode start
    byte_aligned: int  # (debug) readed size (byte aligned value)
    word_aligned: int  # (debug) readed size (word aligned value)

    assert(mode >= 0)
    assert(mode < 14)
    assert(n >= 0)
    dst = bytearray(n)
    s = BitStream(src)

    # special zero bit present in stream?
    if mode >= 7:
        mode -= 7
        zero = True

    for j in range(n):
        # special zero sample
        if zero:
            if s.read_bit() == 0:
                dst[j] = 0x80
                continue

        # read opcode (size)
        size = 0
        offset = s.offset
        for i in range(7 - mode):
            if s.read_bit() == 1:
                size += 1
            else:
                break

        # read data (sample)
        if size == 0:
            # opcode (1 bit), data (mode+1 bits)
            e = (1) + (mode+1)
            sample = sign_extend(s.read_int(mode + 1), mode + 1)
        elif size < 7 - mode:
            # opcode (size+1 bits), data (mode+size bits)
            e = (size+1) + (mode+size)
            sample = unpack_excess(s.read_int(mode + size), mode + size)
        else:
            # opcode (7-mode bits), data (7 bits)
            e = (7-mode) + (7)
            sample = unpack_excess(s.read_int(7), 7)

        # (debug) check opcode size
        #if s.offset - offset != e:
        #    raise ValueError("invalid opcode size: "
        #                     + f"expected {e:d} bits, "
        #                     + f"readed {s.offset-offset:d} bits")

        # store sample
        dst[j] = 0x80 + sample

    # (debug) check that we fully read buffer
    #byte_aligned = (s.offset + 7) // 8
    #word_aligned = (byte_aligned + 1) // 2 * 2
    #if word_aligned != len(src):
    #    raise ValueError("source buffer not readed fully: invalid bitstream?")

    return dst

def decode_lpc(samples: bytearray, factors: ArrayOfInt) -> bytearray:
    nsamples: int = len(samples)
    nfactors: int = len(factors)
    dst: bytearray = bytearray(nfactors)
    accum: int
    i: int
    j: int

    dst += samples
    for i in range(nsamples):
        accum = 0
        for j in range(nfactors):
            accum += sign_extend(dst[i+j] ^ 0x80, 8) * factors[nfactors-j-1]
        accum += 0x800
        dst[i+nfactors] = (dst[i+nfactors] - (accum >> 12) & 0xff) & 0xff

    return dst[nfactors:]

def read_u8snd(path: str) -> tuple[bytearray, int]:
    fp: IO[bytes]       # input file
    result: bytearray   # output buffer
    file_size: int      # full file size
    size: int           # decompressed stream size
    rate: int           # bitrate per second
    ver: int            # unknown header byte (version?)
    hdr: bytes          # unknown header bytes
    fsize: int          # frame size (>= 8)
    fsamples: int       # frame samples
    funk: int           # frame (unknown word, checksum?)
    fmode: int          # frame mode
    buf: bytearray      # decompressed frame samples
    data: bytearray     # frame bytes (include header and size)
    i: int              # 
    checksum: int       # calculated checksum
    factors: ArrayOfInt # LPC factors (array of short)

    with open(path, "rb") as fp:
        # get file size
        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()
        fp.seek(0)
        if file_size < 32:
            raise ValueError("file too small to be U8SND")

        # read header
        [size, rate, ver] = struct.unpack("<LHB", fp.read(7))
        if ver != 1:
            print("warning: version value non-zero", file=sys.stderr)
        hdr = fp.read(25)
        if hdr != b"\0" * 25:
            print("warning: header values non-zero", file=sys.stderr)

        # big file segment table
        if fp.tell() + 2 < file_size:
            [fsize] = struct.unpack("<H", fp.read(2))
            fp.seek(-2, os.SEEK_CUR)
            if fsize == 32 and size > 32767:
                # just skip it
                fp.seek(32 + 256)

        result = bytearray()
        while fp.tell() < file_size:
            # read frame
            [fsize] = struct.unpack("<H", fp.read(2))
            fp.seek(-2, os.SEEK_CUR)
            if fsize < 8:
                raise ValueError("invalid frame size (miniaml is 8)")
            data = bytearray(fp.read(fsize))

            # calc checksum
            checksum = 0
            for i in range(fsize // 2):
                checksum ^= struct.unpack("<H", data[i*2:(i+1)*2])[0]
            if checksum != 0xACED:
                raise ValueError("checksum missmatch")

            # get header values
            [fsize, fsamples, funk, fmode, fskip] = struct.unpack(
                "<HHHBB",
                data[0:8]
            )
            if (fmode <= 7) or (fmode >= 23):
                raise ValueError("unknown frame mode {fmode:d}")

            # decompress entropy
            buf = decode_bitstream(fmode - 8, fsamples, data[8+2*fskip:fsize])
            assert(len(buf) == fsamples)

            # apply LPC (Linear Predictive Conding)
            if fskip > 0:
                factors = ArrayType('h')
                for i in range(fskip):
                    factors.append(
                        struct.unpack("<h", data[8+2*i:8+2*(i+1)])[0]
                    )
                buf = decode_lpc(buf, factors)
                assert(len(buf) == fsamples)

            # save decompressed frame
            result += buf

    if len(result) > size:
        print("warning: decompressed more than mentioned in header"
              + f" ({len(result)} > {size})", file=sys.stderr)
    elif len(result) < size:
        print("warning: decompressed less than mentioned in header"
              + f" ({len(result)} < {size})", file=sys.stderr)

    return result, rate

# ============================================================================ #
#                        Ultima 8 Sound Format (Writer)                        #
# ============================================================================ #

def _compress_samples(src: bytearray, mode: int) -> tuple[bytearray, int]:
    value: int     # source value
    zero: bool     # insert zero bits
    sample: int    # signed value
    code: int      # Excess Code Data
    bits: int      # Excess Code Bits
    size: int      # Opcode (size)
    cost: int      # (debug) Expected bits for opcode
    offset: int    # (debug) Opcode start
    s: BitStream   # bitstream writer

    s = BitStream(bytearray())

    # select mode
    assert(mode >= 0 and mode <= 13)
    zero = False
    if mode >= 7:
        mode -= 7
        zero = True

    for value in src:
        # convert to signed sample
        sample = value - 0x80

        # write special zero-bit
        if zero:
            if sample == 0:
                s.write_bit(0)
                # thats all, compress next value
                continue
            else:
                s.write_bit(1)

        # write opcode + data
        offset = s.offset
        [code, bits] = pack_excess(sample)
        if in_two_complement(sample, mode + 1):
            size = 0
            cost = (1) + (mode+1)
            s.write_bit(0)                         # opcode mode+1
            s.write_int(sample, mode + 1)          # data
        elif bits < 7:
            size = bits - mode
            cost = (size+1) + (mode+size)
            s.write_int((1 << size) - 1, size + 1) # opcode mode+size
            s.write_int(code, bits)                # data
        elif bits == 7:
            size = 7 - mode
            cost = (size) + (7)
            s.write_int((1 << size + 1) - 1, size) # opcode 7
            s.write_int(code, 7)                   # data
        else:
            raise ValueError("failed to select opcode")

        # (debug) check opcode size
        if s.offset - offset != cost:
            raise ValueError("invalid opcode size")

    return s.data, s.offset

def _compress(src: bytearray) -> tuple[bytearray, int]:
    data: bytearray # final buffer
    mode: int       # final mode
    bits: int       # final bits
    tmp: bytearray  # temp buffer
    m: int          # temp mode
    b: int          # temp bits

    # try all modes and select smaller result
    mode = 0
    [data, bits] = _compress_samples(src, mode)
    for m in range(1, 7*2):
        [tmp, b] = _compress_samples(src, m)
        if b < bits:
            data = tmp
            mode = m
            bits = b

    return data, 8 + mode

def _compress_frame(src: bytearray, align: int = 2) -> bytearray:
    head: bytes     # frame header
    data: bytearray # frame data (entropy)
    mode: int       # frame mode
    checksum: int   # frame checksum
    pad: int        # need bytes to align

    # compress data
    assert(align > 0)
    [data, mode] = _compress(src)
    pad = (align - len(data) % align)

    # build header
    head = struct.pack("<HHHBB",
        8 + len(data) + pad, # chunk size
        len(src),            # decompressed size
        0,                   # chksum (fixup later)
        mode,                # mode
        0,                   # num factors
    )

    # build frame
    data = bytearray(head) + data + (b"\0" * pad)

    # compute and fixup checksum
    checksum = 0xACED
    for i in range(len(data) // 2):
        checksum ^= struct.unpack("<H", data[i*2:(i+1)*2])[0]
    struct.pack_into("<H", data, 4, checksum)

    return data

def write_u8snd(path: str, raw: bytearray, rate: int,
                frame: int = 1024, align: int = 2) -> None:
    i: int        # frame id
    size: int     # decompressed size of frame
    offset: int   # curret frame position
    bufsize: int  # size of decompressed buffer
    seg: int      # current segment id 
    segsize: int  # current segment size
    numseg: int   # number of segments
    fp: IO[bytes] #
    start: int    # segment start offset in raw buffer
    fstart: int   # frame start offset in raw buffer
    limit: int    # max bufer size applied by game
    segment: int  # max segment size (constant)
    maxrate: int  # max rate for this sound

    assert(rate > 0 and rate < 65536)
    assert(frame > 32 and frame < 65536)
    assert(align >= 1)
    segment = 0xFE00
    bufsize = len(raw);
    limit = segment * 3 // 2
    numseg = (bufsize + segment - 1) // segment
    maxrate = limit // (bufsize // rate)

    if bufsize > limit:
        print("warning: sound too big, game may crash.", file=sys.stderr)
        print(f"hint: reduce sample rate down to {maxrate}"
              + f" to fit into {limit} byte buffer", file=sys.stderr)

    # not sure, what happens in the game
    # probably there must be fixed structure (2 segments + eof)
    if numseg > (256 - 4 - 8) // 8:
        raise ValueError("segment limit exceed (sound too big)")

    with open(path, "wb") as fp:
        # write header
        fp.write(
            struct.pack("<LHB25s",
                bufsize,   # unpacked size
                rate,      # sample rate per sec
                1,         # version
                b"\0" * 25 # pad
            )
        )

        # big file segment table (fixup later)
        if numseg > 1:
            fp.write(struct.pack("<L252s", 32, b"\0" * 252))

        # write segments (big buffer must be divided by segments)
        for seg in range(numseg):
            start = seg * segment
            segsize = min(bufsize - start, segment)

            # fixup segment table
            if numseg > 1:
                offset = fp.tell()
                fp.seek(32 + 4 + seg*8)
                fp.write(struct.pack("<LL", offset, start + segsize + 32))
                fp.seek(offset)

            # write frames (common size less than segment)
            for i in range((segsize + frame - 1) // frame):
                fstart = start + i * frame
                size = min(segsize - i * frame, frame)
                fp.write(_compress_frame(raw[fstart:fstart+size], align))

        # terminate segment table
        if numseg > 1:
            offset = fp.tell()
            fp.seek(32 + 4 + numseg*8)
            fp.write(struct.pack("<LL", offset, 0))
            fp.seek(offset)

# ============================================================================ #
#                             Microsoft RIFF WAVE                              #
# ============================================================================ #

def write_wave(path: str, raw: bytearray, rate: int) -> None:
    fp: IO[bytes]

    with open(path, "wb") as fp:
        # RIFF header
        fp.write(
            struct.pack("<4sL4s",
                b"RIFF",       # signature
                len(raw) + 36, # data size
                b"WAVE"        # subformat
            )
        )

        # WAVE FMT chunk
        fp.write(
            struct.pack("<4sLHHLLHH",
                b"fmt ",       # chunk type
                16,            # chunk size
                1,             # audio format -> PCM
                1,             # NumChannels -> MONO
                rate,          # SampleRate
                rate,          # = SampleRate * NumChannels * BitsPerSample / 8
                1,             # = NumChannels * BitsPerSample / 8
                8              # BitsPerSample -> 8 bit (unsigned)
            )
        )

        # WAVE DATA chunk
        fp.write(
            struct.pack("<4sL",
                b"data",       # chunk type
                len(raw)       # chunk size
            )
        )
        fp.write(raw)

def read_wave(path: str) -> tuple[bytearray, int]:
    fp: IO[bytes]
    filesize: int
    riffsize: int
    size: int
    fmt: int
    chan: int
    rate: int
    byterate: int
    align: int
    bits: int
    data: bytearray

    with open(path, "rb") as fp:
        # check size
        fp.seek(0, os.SEEK_END)
        filesize = fp.tell()
        fp.seek(0)
        if filesize < 44:
            raise ValueError("too small to be RIFF WAVE")

        # read RIFF-WAVE header
        if fp.read(4) != b"RIFF":
            raise ValueError("not RIFF file")
        riffsize = struct.unpack("<L", fp.read(4))[0]
        if riffsize < 36:
            raise ValueError("too small to be RIFF WAVE")
        if riffsize + 8 > filesize:
            raise ValueError("file too small (truncated)")
        if fp.read(4) != b"WAVE":
            raise ValueError("not RIFF WAVE")

        # read FMT chunk
        if fp.read(4) != b"fmt ":
            raise ValueError("extected WAVE-fmt chunk")
        size = struct.unpack("<L", fp.read(4))[0]
        if (size < 16) or (fp.tell() + size > 8 + riffsize):
            raise ValueError("invalid WAVE-fmt size")
        [fmt, chan, rate, byterate, align, bits] = struct.unpack_from(
            "<HHLLHH",
            fp.read(size)
        )
        if fmt != 1:
            raise ValueError(f"expected PCM fromat (readed {fmt:d})")
        if chan != 1:
            raise ValueError(f"expected MONO sound (readed {chan:d})")
        if bits != 8:
            raise ValueError(f"expected U8 PCM format (readed {bits:d}-bit)")
        if byterate != rate * chan * bits / 8:
            raise ValueError("invalid byte rate field")
        if align != chan * bits / 8:
            raise ValueError("invalid align field")

        # read DATA chunk
        if fp.read(4) != b"data":
            raise ValueError("extected WAVE-data chunk")
        size = struct.unpack("<L", fp.read(4))[0]
        if fp.tell() + size > 8 + riffsize:
            raise ValueError("invalid WAVE-data size")
        data = bytearray(fp.read(size))

    return data, rate

# ============================================================================ #
#                                     Main                                     #
# ============================================================================ #

def _help() -> None:
    fp: IO[str] = sys.stderr
    print("Usage: soundconv <src_fmt> <dst_fmt> <src> <dst>", file=fp)
    print("Formats:", file=fp)
    print("    wav   - Microsoft RIFF WAVE (PCM U8 MONO Subset)", file=fp)
    print("    u8snd - Ultima 8 Sound Format", file=fp)
    exit(1)

def _main() -> int:
    rate: int
    data: bytearray
    smode: str
    dmode: str
    pfn: str

    if len(sys.argv) <= 4:
        _help()
    smode = sys.argv[1]
    dmode = sys.argv[2]
    src = sys.argv[3]
    dst = sys.argv[4]

    if smode not in ["wav", "u8snd"]:
        print("error: unknown source format " + smode, file=sys.stderr)
        _help()
    if dmode not in ["wav", "u8snd"]:
        print("error: unknown destination format " + dmode, file=sys.stderr)
        _help()

    try:
        pfn = src
        if smode == "wav":
            [data, rate] = read_wave(src)
        elif smode == "u8snd":
            [data, rate] = read_u8snd(src)

        pfn = dst
        if dmode == "wav":
            write_wave(dst, data, rate)
        elif dmode == "u8snd":
            write_u8snd(dst, data, rate, frame=1024, align=2)
    except (ValueError, OSError) as e:
        print("error: " + pfn + ": " + str(e), file=sys.stderr)
        exit(1)

    return 0

if __name__ == "__main__":
    exit(_main())
