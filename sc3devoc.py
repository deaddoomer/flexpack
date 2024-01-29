#!/usr/bin/python3

"""
Voice archive (un)packer for Star Control 3
"""

from __future__ import annotations # Python 3.7+

import io
import sys
import struct

from typing import IO, Union
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterator
from configparser import ConfigParser

__author__ = "DeaDDooMER"
__license__ = "MIT"
__version__ = "1.0.1"

class BitStream:
    data: bytearray
    offset: int

    def __init__(self, data: bytearray, offset: int = 0) -> None:
        assert(offset >= 0)
        assert(offset <= len(data) * 8)
        self.data = data
        self.offset = offset

    def __len__(self) -> int:
        return len(self.data) * 8

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

class VocArchive(ABC):
    # for use as `value = voc[key]`
    @abstractmethod
    def __getitem__(self, key: int) -> FDesc:
        ...

    # for use as `voc[key] = value`
    @abstractmethod
    def __setitem__(self, key: int, value: FDesc) -> None:
        ...

    # for use as `for i in voc:`
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        ...

class FDesc:
    offset: int
    flags: int
    csize: int
    usize: int
    fp: IO[bytes]

    def __init__(self, i: int, offset: int, fp: IO[bytes]) -> None:
        self.i = i
        self.offset = offset
        self.frame = -1
        self.csize = -1
        self.usize = -1
        self.fp = fp

class VocArchiveBin(VocArchive):
    tab: dict[int,FDesc]

    def __init__(self, fp: IO[bytes]) -> None:
        self.tab = {}

    def __getitem__(self, key: int) -> FDesc:
        return self.tab[key]

    def __setitem__(self, key: int, value: FDesc) -> None:
        self.tab[key] = value

    def __iter__(self) -> Iterator[int]:
        return iter(self.tab)

class FInfo:
    i: int
    path: str
    frame: int
    offset: int

    def __init__(self, i: int, path: str, frame: int) -> None:
        self.i = i
        self.path = path
        self.frame = frame
        self.offset = 0


def sign_extend(value: int, bits: int) -> int:
    # Convert two's complement value to python integer
    assert(value >= 0)
    assert(bits >= 0)
    sign_bit: int = 1 << bits - 1
    return (value & sign_bit - 1) - (value & sign_bit)

def clamp(value: int, vmin: int, vmax: int) -> int:
    return max(min(value, vmax), vmin)

def sign_iex(val: int, bits: int) -> int:
    assert(val >= 0)
    assert(bits > 0)
    b: int = bits - 1
    mask: int = (1 << b) - 1
    sign: int = ~val >> b & 1
    return (val & mask) - (sign << b) + (sign ^ 1)

# ============================================================================ #
#                   Star Control 3 VOC Archive Decompressor                    #
# ============================================================================ #

TAB = [
      -8,   -7,  -6,  -5,  -4,  -3,  -2,  -1,  1,  2,  3,  4,  5,  6,   7,   8,
     -16,  -14, -12, -10,  -8,  -6,  -4,  -2,  2,  4,  6,  8, 10, 12,  14,  16,
     -24,  -21, -18, -15, -12,  -9,  -6,  -3,  3,  6,  9, 12, 15, 18,  21,  24,
     -32,  -28, -24, -20, -16, -12,  -8,  -4,  4,  8, 12, 16, 20, 24,  28,  32,
     -40,  -35, -30, -25, -20, -15, -10,  -5,  5, 10, 15, 20, 25, 30,  35,  40,
     -48,  -42, -36, -30, -24, -18, -12,  -6,  6, 12, 18, 24, 30, 36,  42,  48,
     -56,  -49, -42, -35, -28, -21, -14,  -7,  7, 14, 21, 28, 35, 42,  49,  56,
     -64,  -56, -48, -40, -32, -24, -16,  -8,  8, 16, 24, 32, 40, 48,  56,  64,
     -72,  -63, -54, -45, -36, -27, -18,  -9,  9, 18, 27, 36, 45, 54,  63,  72,
     -80,  -70, -60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60,  70,  80,
     -88,  -77, -66, -55, -44, -33, -22, -11, 11, 22, 33, 44, 55, 66,  77,  88,
     -96,  -84, -72, -60, -48, -36, -24, -12, 12, 24, 36, 48, 60, 72,  84,  96,
    -104,  -91, -78, -65, -52, -39, -26, -13, 13, 26, 39, 52, 65, 78,  91, 104,
    -112,  -98, -84, -70, -56, -42, -28, -14, 14, 28, 42, 56, 70, 84,  98, 112,
    -120, -105, -90, -75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120,
    -127, -112, -96, -80, -64, -48, -32, -16, 16, 32, 48, 64, 80, 96, 112, 127,
]
TABMODE = False

def decode(m: int, c: int, s: int) -> int:
    assert(m >= 1 and m <= 16)
    assert(c >= 0 and c < (1 << s))
    assert(s in [1, 2, 4])
    return clamp(sign_iex(c, s) * m, -127, 127)

def decompress(src: bytearray, usize: int, frame: int) -> bytearray:
    i: int
    j: int
    k: int
    c: int

    s: int
    t0: int
    t1: int
    sample: int

    op: int
    _c: int
    _n: int
    _s: int
    _m: int

    rd: BitStream
    dst: bytearray

    assert(usize >= 0)
    assert(frame > 0)

    i = 0
    sample = 0x80
    dst = bytearray(usize)
    rd = BitStream(src, 0)
    while rd.offset < len(rd):
        # get opcode
        assert(rd.offset % 8 == 0)
        op = rd.read_int(8)
        if op & 0x80 != 0:
            # 7 bit literal
            _c = op & 0x7F
            sample = _c << 1
            dst[i] = sample
            i += 1
        elif op & 0x40 != 0:
            # repeat last sample N times
            _n = op & 0x3F
            for j in range(_n + 1):
                dst[i] = sample
                i += 1
        else:
            # frame of relative samples
            _s = (op >> 4) & 3
            _m = op & 0xF

            j = _m << 4
            if _s == 1:
                s = 1
                j += 7
            elif _s == 2:
                s = 2
                j += 6
            elif _s == 3:
                s = 4
            else: # _s == 0:
                raise ValueError("invalid bitstream")

            t0 = 0
            t1 = 0
            for k in range(frame):
                c = rd.read_int(s)
                t1 = t0
                if TABMODE:
                    t0 = TAB[j+c]
                else:
                    t0 = decode(_m + 1, c, s)
                sample = clamp(sample + t0, 0, 255)

                if abs(t0) == _m + 1 and t0 == -t1:
                    dst[i-1] = clamp(dst[i-1] + (-t1 - (t0 >> 7) >> 1), 0, 255)

                dst[i] = sample
                i += 1

    return dst

# ============================================================================ #
#                      Star Control 3 VOC Archive Reader                       #
# ============================================================================ #

def read_voc(fp: IO[bytes]) -> VocArchive:
    fsize: int         # archive size
    count: int         # entries in table
    offset: int        #
    i: int             #
    hdr: bytes         # RIFF/WAVE header
    fd: FDesc          # current file in archive
    voc: VocArchiveBin #

    # get archive size
    voc = VocArchiveBin(fp)
    fp.seek(0, io.SEEK_END)
    fsize = fp.tell()
    fp.seek(0)

    # read table of sounds
    [count] = struct.unpack("<H", fp.read(2))
    for i in range(count - 1):
        [offset] = struct.unpack("<L", fp.read(4))
        if offset != 0:
            voc[i+1] = FDesc(i+1, offset, fp)

    # read entry data
    for i in voc:
        fd = voc[i]
        fp.seek(fd.offset)
        [fd.frame, fd.csize] = struct.unpack("<HL", fp.read(6))
        hdr = fp.read(44)
        [fd.usize] = struct.unpack_from("<L", hdr, 40)

    return voc

def getname(inipath: str, i: int) -> str:
    return str(Path(Path(inipath).name).with_suffix(f".{i:04d}.wav"))

def write_desc(voc: VocArchive, path: str) -> None:
    fp: IO[str]
    fd: FDesc
    i: int

    with open(path, 'w') as fp:
        for i in voc:
            fd = voc[i]
            print(f"[{i}]", file=fp)
            print(f"frame={fd.frame}", file=fp)
            print(f"file={getname(path, i)}", file=fp)
            print(file=fp)

def extract_voc(voc: VocArchive, path: str) -> None:
    fd: FDesc
    hdr: bytes
    data: bytes
    cdata: bytearray
    fp: IO[bytes]
    name: str
    i: int
    n: int

    # max id
    n = 0
    for i in voc:
        n = max(i, n)

    for i in voc:
        print(f"{i+1:4d} / {n + 1}")

        # read data & decompress it
        fd = voc[i]
        fd.fp.seek(fd.offset + 6)
        hdr = fd.fp.read(44)
        if fd.frame == 0:
            data = fd.fp.read(fd.usize)
        else:
            cdata = bytearray(fd.fp.read(fd.csize))
            data = decompress(cdata, fd.usize, fd.frame)

        # write decompressed WAV
        name = str(Path(path).parent / getname(path, i))
        with open(name, 'wb') as fp:
            fp.write(hdr)
            fp.write(data)

# ============================================================================ #
#                      Star Control 3 VOC Archive Writer                       #
# ============================================================================ #

def read_desc(path: str) -> list[FInfo]:
    ini: ConfigParser
    s: str
    i: int
    frame: int
    tab: list[FInfo]
    name: str

    ini = ConfigParser()
    ini.read(path)

    tab = []
    for s in ini.sections():
        if s.isdigit():
            if ini.has_option(s, "file"):
                i = int(s)
                if i >= 1 and i < 65535:
                    frame = clamp(ini.getint(s, "frame", fallback=0), 0, 65535)
                    name = str(Path(path).parent / ini[s]["file"])
                    tab.append(FInfo(i, name, frame))
                else:
                    raise ValueError(f"invalid object id {s}")
            else:
                raise ValueError(f"object {s} has no file path")

    return tab

def get_wave(path: str) -> bytes:
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
    hdr: bytes
    data: bytes

    with open(path, "rb") as fp:
        # check size
        fp.seek(0, io.SEEK_END)
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
            print(f"warning: file {path} truncated "
                  + f"({riffsize + 8 - filesize} bytes)", file=sys.stderr)
            #raise ValueError("file too small (truncated)")
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
        if bits not in [8, 16]:
            raise ValueError(f"expected U8/S16 PCM format (readed {bits:d}-bit)")
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
        data = fp.read(size)

    # Star Control 3 requires fixed-size header
    hdr = struct.pack(
        "<4sL4s 4sL HHLLHH 4sL",
        b"RIFF",            # RIFF signature
        44 + len(data) - 8, # RIFF size
        b"WAVE",            # WAVE signature
        b"fmt ",            # fmt header
        16,                 # fmt 16 bytes
        fmt, chan, rate, byterate, align, bits,
        b"data",            # data header
        len(data)           # data size
    )

    return hdr + data

def write_pad(fp: IO[bytes], skip: int, align: int) -> None:
    pad: int

    assert(skip >= 0)
    assert(align >= 1)
    pad = skip + (fp.tell() + align - 1) // align * align - fp.tell()
    fp.write(b"\0" * pad)

def write_voc(path: str, lst: list[FInfo]) -> None:
    ALIGN: int = 4096
    count: int
    info: FInfo
    slst: list[FInfo]
    fp: IO[bytes]
    data: bytes

    slst = lst.copy()
    slst.sort(key=lambda i: i.i)
    count = slst[-1].i

    with open(path, 'wb') as fp:
        # write table (fixup later)
        fp.write(struct.pack("<H", count))
        write_pad(fp, 4*count, ALIGN)

        # write data
        for info in lst:
            info.offset = fp.tell()
            try:
                data = get_wave(info.path)
            except (ValueError, OSError) as e:
                print("error: " + info.path + ": " + str(e), file=sys.stderr)
                exit(1)
            #fp.write(struct.pack("<HL", info.frame, len(data) - 44))
            fp.write(struct.pack("<HL", 0, len(data) - 44))
            fp.write(data)
            write_pad(fp, 0, ALIGN)

        # fixup table
        for info in slst:
            fp.seek(2 + (info.i - 1) * 4)
            fp.write(struct.pack("<L", info.offset))

# ============================================================================ #
#                                     Main                                     #
# ============================================================================ #

def help(exitcode: int) -> None:
    fp: IO[str] = sys.stderr
    print("Usage: unvoc create  <src.ini> <dst.voc>", file=fp)
    print("       unvoc extract <src.voc> <dst.ini>", file=fp)
    print("       unvoc getdesc <src.voc> <dst.ini>", file=fp)
    print("       unvoc help", file=fp)
    exit(exitcode)

def main(argc: int, argv: list[str]) -> int:
    fp: IO[bytes]
    voc: VocArchive
    lst: list[FInfo]
    cmd: str
    src: str
    dst: str

    if argc <= 1:
        help(1)

    cmd = argv[1]
    if cmd == "help":
        help(0)
    elif cmd in ["getdesc", "extract"]:
        if argc <= 3:
            help(1)
        src = argv[2]
        dst = argv[3]
        try:
            with open(src, 'rb') as fp:
                voc = read_voc(fp)
                write_desc(voc, dst)
                if argv[1] == "extract":
                    extract_voc(voc, dst)
        except (ValueError, OSError) as e:
            print("error: " + src + ": " + str(e), file=sys.stderr)
            exit(1)
    elif cmd in ["create"]:
        if argc <= 3:
            help(1)
        src = argv[2]
        dst = argv[3]

        try:
            lst = read_desc(src)
        except (ValueError, OSError) as e:
            print("error: " + src + ": " + str(e), file=sys.stderr)
            exit(1)

        try:
            write_voc(dst, lst)
        except (ValueError, OSError) as e:
            print("error: " + dst + ": " + str(e), file=sys.stderr)
            exit(1)
    else:
        help(1)

    return 0

if __name__ == "__main__":
    exit(main(len(sys.argv), sys.argv))