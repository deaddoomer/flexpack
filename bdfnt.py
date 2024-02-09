#!/usr/bin/env python3

from __future__ import annotations # Python 3.7+

import io
import sys
import struct
import zlib

from typing import IO, Union, Optional
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser, Namespace

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

@dataclass
class LetterDesc:
    offset: int
    w: int
    h: int
    x: int
    y: int

@dataclass
class Font:
    bg: int              # background color
    size: int            # ascent + descent height
    baseline: int        # letter baseline
    tracking: int        # inter-letter spacing
    leading: int         # inter-line spacing
    space: int           # space width
    glyph: list[Image]   # letter bitmaps

class Image:
    bitmap: bytearray
    w: int
    h: int

    def __init__(self, w: int, h: int, bitmap: Optional[bytearray] = None) -> None:
        assert(w >= 0)
        assert(h >= 0)
        assert(bitmap is None or len(bitmap) == w * h)
        self.w = w
        self.h = h
        if bitmap is None:
            self.bitmap = bytearray(w * h)
        else:
            self.bitmap = bitmap

    def clear(self, color: int) -> None:
        i: int
        assert(color >= 0 and color <= 255)
        for i in range(len(self.bitmap)):
            self.bitmap[i] = color

    def copy(self, x0: int, y0: int,
             src: Image, x: int, y: int, w: int, h: int) -> None:
        j: int
        dline: int
        sline: int
        assert(src is not self)
        assert(w >= 0)
        assert(h >= 0)
        assert(w <= max(self.w - x0, src.w - x))
        assert(h <= max(self.h - y0, src.h - y))
        for j in range(h):
            sline = (y + j) * src.w + x
            dline = (y0 + j) * self.w + x0
            self.bitmap[dline:dline+w] = src.bitmap[sline:sline+w]

    def get(self, x: int, y: int) -> int:
        assert(x >= 0 and x < self.w)
        assert(y >= 0 and y < self.h)
        return self.bitmap[y*self.w+x]

# ============================================================================ #
#                       Portable Network Graphics Format                       #
# ============================================================================ #

PNG_8_BPP: int = 8
PNG_INDEXED: int = 3
PNG_DEFLATE: int = 0
PNG_FILTER_NONE: int = 0
PNG_INTERLANCE_NONE: int = 0

def write_chunk(fp: IO[bytes], typ: bytes, data: bytes,
                compress: bool=False) -> None:
    crc: int
    zdata: bytes
    c: zlib._Compress

    # compress
    if compress:
        c = zlib.compressobj(level=9, wbits=15)
        zdata = c.compress(data)
        zdata += c.flush()
    else:
        zdata = data

    # compute crc
    crc = zlib.crc32(typ)
    crc = zlib.crc32(zdata, crc)

    # write chunk
    fp.write(struct.pack(">L", len(zdata)))
    fp.write(typ)
    fp.write(zdata)
    fp.write(struct.pack(">L", crc))

def read_chunk(fp: IO[bytes], decompress: bool=False) -> tuple[bytes, bytes]:
    crc: int
    ecrc: int
    size: int
    typ: bytes
    data: bytes
    zdata: bytes

    # read chunk
    size = struct.unpack(">L", fp.read(4))[0]
    typ = fp.read(4)
    zdata = fp.read(size)
    ecrc = struct.unpack(">L", fp.read(4))[0]

    # check crc
    crc = zlib.crc32(typ)
    crc = zlib.crc32(zdata, crc)
    if crc != ecrc:
        raise ValueError("CRC missmatch")

    # decompress it
    if decompress:
        data = zlib.decompress(zdata, wbits=15)
    else:
        data = zdata

    return typ, data

def write_png(name: str, w: int, h: int, x: int, y: int,
              bits: bytearray, pal: bytearray) -> None:
    i: int         # scanline counter
    fp: IO[bytes]  # dst file
    raw: bytearray # bitmap with filter bytes

    with open(name, "wb") as fp:
        # write PNG header
        fp.write(b"\x89PNG\r\n\x1A\n")

        # write basic metadata
        write_chunk(fp, b"IHDR",
            struct.pack(
                ">LLBBBBB",
                w, h, PNG_8_BPP,
                PNG_INDEXED, PNG_DEFLATE,
                PNG_FILTER_NONE, PNG_INTERLANCE_NONE
            )
        )

        # write palette
        write_chunk(fp, b"PLTE", pal)

        # Mark color 255 as trasparent
        # BUG: GIMP saves transparency back as color 0
        #write_chunk(fp, b"tRNS", b"\xFF" * 255 + b"\x00")

        # Write offset
        # BUG: GIMP didn't draw image when offset applied
        #write_chunk(fp, b"oFFs", struct.pack(">llB", x, y, 0))

        # add filter type to every scanline (as required by specification)
        raw = bytearray((w + 1) * h)
        for i in range(h):
            raw[(w+1)*i] = 0
            raw[(w+1)*i+1:(w+1)*i+1+w] = bits[w*i:w*i+w]

        # write bitmap
        write_chunk(fp, b"IDAT", raw, compress=True)

        # write end
        write_chunk(fp, b"IEND", b"")

def read_png(name: str) -> Image:
    i: int             # scanline counter
    w: int             # image width
    h: int             # image height
    typ: bytes         # chunk type
    data: bytes        # decompressed data
    a: tuple[int, ...] # [w, h, depth, pixel_fmt, comp, filter, interlance]
    bitmap: bytearray  # bitmap WxH
    fp: IO[bytes]      # src file

    with open(name, "rb") as fp:
        # read header
        if fp.read(8) != b"\x89PNG\r\n\x1A\n":
            raise ValueError("invalid PNG header")

        # read basic metadata
        [typ, data] = read_chunk(fp)
        if typ != b"IHDR":
            raise ValueError("missing IHDR chunk")
        if len(data) != 13:
            raise ValueError("invalid IHDR size " + str(len(data)))
        a = struct.unpack(">LLBBBBB", data);
        w = a[0]
        h = a[1]

        # check metadata
        if w * h == 0:
            raise ValueError("invalid image size " + str(w) + "x" + str(h))
        if a[2] != PNG_8_BPP:
            raise ValueError("unsupported depth " + str(a[2]))
        if a[3] != PNG_INDEXED:
            raise ValueError("unsupported bitmap format " + str(a[3]))
        if a[4] != PNG_DEFLATE:
            raise ValueError("unsupported compression method " + str(a[4]))
        if a[5] != PNG_FILTER_NONE:
            raise ValueError("unsupported filter method " + str(a[5]))
        if a[6] != PNG_INTERLANCE_NONE:
            raise ValueError("unsupported interlance method " + str(a[6]))

        # read chunks, ignore everything except IDAT
        i = 0
        bitmap = bytearray(w * h)
        [typ, data] = read_chunk(fp)
        while typ != b"IEND":
            if typ == b"IDAT":
                # remove filter byte and copy
                data = zlib.decompress(data, wbits=15)
                for j in range(len(data) // (w + 1)):
                    if i >= h:
                        raise ValueError("too many scanlines")
                    bitmap[w*i:w*(i+1)] = data[1+i+w*i:1+i+w*(i+1)]
                    i += 1
            [typ, data] = read_chunk(fp)

        return Image(w, h, bitmap)

# ============================================================================ #
#                           Black Dahlia Font Format                           #
# ============================================================================ #

def read_font(path: str) -> Font:
    fp: IO[bytes]         #
                          # --- HEADER 1 ---
    m: int                # first char id in font
    n: int                # N bitmaps in font
    key_color: int        # transparent color id
    lf_height: int        # height for char 0xA (linefeed)
    u2: int               # font base (unused?)
    interval: int         # inverval width between letters
    lf_interval: int      # interval heigth between lines
    space_width: int      # width of space char 0x20
    letter: dict[int,LetterDesc] = {} # letter metrics and offset
                          # --- HEADER 2 ---
    flags: int            #
    size: int             # size of dictionary
    palsize: int          # size of palette
    unk0: int             # transparent color in dictionary (not used)
    comp: int             # compression type
    mtab: bytes           # offset table
    aa: int               # size of uncompressed data (converted via dictionary)
    w: int                # size of bitmap (first)
    h: int                # size of bitmap (second)
    dd: int               # size of compressed data
                          # --- HEADER 3 ---
    a: int                # size of uncompressed data (converted via dictionary)
    b: int                # size of compressed data
    is_mcg: bool          # MCG header readed

    extmode: bool         # use 0xE opcode as RLE
    s: BitStream          # bitstream reader
    src: bytearray        # compressed data
    bitmap: bytearray     # decompressed data
    dictionary: bytearray # dictionary for comressed data
    i: int                #
    j: int                # decompressor counter
    op: int               # opcode
    rs: int               # repeat length
    ds: int               # repeat byte
    prev: int             # previous dict index

    bmp: bytearray        # glyph data
    img0: Image           # glyph image
    img1: Image           # glyph image with applied offsets
    lst: list[Image]      # list of glyphs

    with open(path, 'rb') as fp:
        # read HEADER 1
        if fp.read(4) != b"NF2T":
            raise ValueError("not font")
        [m, n, key_color, lf_height, u2, interval, lf_interval, space_width] = struct.unpack("<BBBBBBBB", fp.read(8))

        # read metrics
        for i in range(n):
            letter[m+i] = LetterDesc(*struct.unpack("<LBBBB", fp.read(8)))

        # read HEADER 2
        [flags, size, palsize, unk0, comp, mtab, aa, w, h, dd] = struct.unpack(
            "<BBBBB15sHHHH", fp.read(28)
        )

        # read HEADER 3 (optional)
        if comp in [4, 5, 6]:
            if fp.read(4) != b"GCMS":
                raise ValueError("invalid svga mcg file")
            [a, b] = struct.unpack("<LL", fp.read(8))
            is_mcg = True
        else:
            a = aa
            b = dd
            is_mcg = False

        # read compressed data
        src = bytearray(fp.read(b))

        # read color table
        if flags & 1 != 0:
            fp.read((palsize + 1) * 3)

        # read dictionary
        if flags & 2 != 0:
            dictionary = bytearray(fp.read(size + 1))
        else:
            dictionary = bytearray(256)
            for i in range(256):
                dictionary[i] = i

        # check compression type
        if comp not in [1, 2] and is_mcg == False:
            raise ValueError("invalid compression type")
        extmode = comp == 2 or comp == 6

        # decompress
        j = 1
        s = BitStream(src)
        prev = s.read_int(8)
        bitmap = bytearray(w * h)
        bitmap[0] = dictionary[prev]
        while j < w * h - a:
            op = s.read_int(4)
            if op == 0xF:
                prev = s.read_int(4)
                bitmap[j] = dictionary[prev]
                j += 1
            elif op == 0xE and extmode:
                rs = s.read_int(8)
                if rs == 0xFF:
                    rs = rs * 256 + s.read_int(8)
                rs = (rs + 2) & 0xFFFF
                ds = bitmap[j-1]
                for i in range(rs):
                    bitmap[j] = ds
                    j += 1
            else:
                prev += mtab[op]
                if prev >= size + 1:
                    prev -= size + 1
                bitmap[j] = dictionary[prev]
                j += 1
        if j != w * h - a:
            raise ValueError("invalid bitstream")

        # process uncompressed data
        for i in range(a, 0, -1):
            prev = s.read_int(8)
            bitmap[j] = dictionary[prev]
            j += 1
        if j != w * h:
            raise ValueError("invalid bitstream")

        # get glyphs
        lst = []
        for i in range(256):
            if i in letter and letter[i].w != 0 and letter[i].h != 0:
                bmp = bitmap[letter[i].offset:letter[i].offset+letter[i].w*letter[i].h]
                img0 = Image(letter[i].w, letter[i].h, bmp)
                img1 = Image(letter[i].x + img0.w, letter[i].y + img0.h)
                img1.clear(key_color)
                img1.copy(letter[i].x, letter[i].y, img0, 0, 0, img0.w, img0.h)
            else:
                img1 = Image(0, 0)
            lst.append(img1)

        # construct font object
        return Font(
                bg=key_color,
                size=lf_height,
                baseline=u2,
                tracking=interval,
                leading=lf_interval,
                space=space_width,
                glyph=lst)

def write_font(name: str, font: Font) -> None:
    fp: IO[bytes]
    i: int
    first: int
    nglyphs: int
    offset: int
    exthdr: bool
    comp: int
    letter: dict[int,LetterDesc] = {}
    data: bytearray = bytearray()
    mtab: bytearray = bytearray(15)

    for i in range(0,256):
        if font.glyph[i].w > 0 and font.glyph[i].h > 0:
            break
    first = i

    for i in range(255,-1,-1):
        if font.glyph[i].w > 0 and font.glyph[i].h > 0:
            break
    nglyphs = i - first + 1

    for i in range(first, first + nglyphs):
        if font.glyph[i].w > 0 and font.glyph[i].h > 0:
            offset = len(data)
        else:
            offset = 0
        letter[i] = LetterDesc(offset, font.glyph[i].w, font.glyph[i].h, 0, 0)
        data += font.glyph[i].bitmap

    if len(data) > 65535:
        raise ValueError("data too big")

    with open(name, "wb") as fp:
        fp.write(
            struct.pack("<4sBBBBBBBB",
                b"NF2T",         # font siganture NF2T
                first,           # first glyph in file
                nglyphs,         # count glyph in file
                font.bg,         # transparent color
                font.size,       # size
                font.baseline,   # baseline
                font.tracking,   # tracking
                font.leading,    # leading
                font.space       # space width
            )
        )
        for i in letter:
            fp.write(
                struct.pack("<LBBBB",
                    letter[i].offset,
                    letter[i].w,
                    letter[i].h,
                    letter[i].x,
                    letter[i].y,
                )
            )
        fp.write(
            struct.pack(
                "<BBBBB15sHHHH",
                0,               # flags: no palette, no dictionary
                255,             # size of dictionary (unused)
                255,             # size of palette (unused)
                font.bg,         # transparent color in dictionary (unused)
                1,               # compression mode 1
                mtab,            # offset table (unused)
                len(data)-1,     # size of uncompressed data in strean
                1,               # width
                len(data),       # height
                len(data),       # size of compressed data stream
            )
        )
        fp.write(data)

# ============================================================================ #
#                                   INI font                                   #
# ============================================================================ #

def write_inifont(name: str, font: Font) -> None:
    pal: bytearray
    img: Image
    fp: IO[str]
    i: int
    nm: str
    fnm: str

    pal = bytearray(b"\x00" * (3*255) + b"\xFF" * 3)
    with open(name, "w") as fp:
        print(f"[FONT]", file=fp)
        print(f"background={font.bg}", file=fp)
        print(f"size={font.size}", file=fp)
        print(f"baseline={font.baseline}", file=fp)
        print(f"tracking={font.tracking}", file=fp)
        print(f"leading={font.leading}", file=fp)
        print(f"space={font.space}", file=fp)

        for i in range(len(font.glyph)):
            img = font.glyph[i]
            if img.w > 0 and img.h > 0:
                nm = Path(name).stem + "." + str(i) + ".PNG"
                fnm = str(Path(name).parent / nm)
                print(file=fp)
                print(f"[{i}]", file=fp)
                print(f"file={nm}", file=fp)
                write_png(fnm, img.w, img.h, 0, 0, img.bitmap, pal)

def cgetint(cfg: ConfigParser, section: str, option: str,
            imin: int, imax: int) -> int:
    assert(imin <= imax)
    value: int = cfg.getint(section, option)
    if value < imin or value > imax:
        raise ValueError(f"Value of {section}.{option} "
                         + f"must be in range [{imin}..{imax}]")
    return value

def read_inifont(name: str) -> Font:
    cfg: ConfigParser  # font ini
    bg: int            #
    size: int          #
    baseline: int      #
    tracking: int      #
    leading: int       #
    space: int         #
    i: int             #
    nm: str            # file name
    fnm: str           # full file name
    lst: list[Image]   # list of glyphs

    # read metrics
    cfg = ConfigParser()
    cfg.read(name)
    if cfg.has_section("FONT") == False:
        raise ValueError("Not ini font")
    bg = cgetint(cfg, "FONT", "background", 0, 255)
    size = cgetint(cfg, "FONT", "size", 0, 255)
    baseline = cgetint(cfg, "FONT", "baseline", 0, 255)
    tracking = cgetint(cfg, "FONT", "tracking", 0, 255)
    leading = cgetint(cfg, "FONT", "leading", 0, 255)
    space = cgetint(cfg, "FONT", "space", 0, 255)

    # read glyphs
    lst = []
    for i in range(256):
        if cfg.has_section(str(i)) and cfg.has_option(str(i), "file"):
            nm = cfg.get(str(i), "file")
            fnm = str(Path(name).parent / nm)
            lst.append(read_png(fnm))
        else:
            lst.append(Image(0, 0))

    # construct font object
    return Font(
            bg=bg,
            size=size,
            baseline=baseline,
            tracking=tracking,
            leading=leading,
            space=space,
            glyph=lst)

# ============================================================================ #
#                                     Main                                     #
# ============================================================================ #

def main() -> int:
    ap: ArgumentParser
    av: Namespace
    modes: list[str] = ["fnt", "ini"]
    font: Font

    ap = ArgumentParser(prog="bdfnt", description="Black Dahlia font converter")
    ap.add_argument("src", help="source file")
    ap.add_argument("dst", help="destination file")
    ap.add_argument("-f", "--from", dest="smode", choices=modes, required=True, help="convert from")
    ap.add_argument("-t", "--to", dest="dmode", choices=modes, required=True, help="convert to")
    av = ap.parse_args()

    if av.smode == "fnt":
        font = read_font(av.src)
    elif av.smode == "ini":
        font = read_inifont(av.src)
    else:
        raise

    if av.dmode == "fnt":
        write_font(av.dst, font)
    elif av.dmode == "ini":
        write_inifont(av.dst, font)
    else:
        raise

    return 0

if __name__ == "__main__":
    exit(main())