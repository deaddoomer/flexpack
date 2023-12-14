#!/usr/bin/env python3

"""
Graphics converter for Crusader: No Remorse aimed to replace game resources.
"""

import os
import sys
import struct
import zlib
import json
import typing

from dataclasses import dataclass
from typing import IO, Optional, Union, SupportsIndex, Any
from pathlib import Path

__author__ = "DeaDDooMER"
__license__ = "MIT"
__version__ = "1.0.0"

class ShapeError(ValueError):
    def __init__(self, msg: str) -> None:
        self.msg = msg

class Frame:
    offset: int        # Offset in file (internal use only)
    size: int          # Size in file (internal use only)
    shape: int         # Shape ID
    frame: int         # Frame ID
    flags: int         # Unknown value
    comp: int          # Compression type (-1 -> auto)
    w: int             # Bitmap Width
    h: int             # Bitmap Height
    x: int             # Offset X
    y: int             # Offset Y
    bitmap: bytearray  # Bitmap WxH (decompressed)

    def __init__(self, shape: int, frame: int, flags: int, comp: int,
                 w: int, h: int, x: int, y: int, bitmap: bytearray,
                 offset: int = -1, size: int = -1) -> None:
        assert(len(bitmap) == w * h)
        self.shape = shape
        self.frame = frame
        self.flags = flags
        self.comp = comp
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.bitmap = bitmap
        self.offset = offset
        self.size = size

class FrameRef:
    frm: Frame  # Reference to frame (can be non unique)
    flags: int  # Unknown flags

    def __init__(self, frame: Frame, flags: int) -> None:
        self.frm = frame
        self.flags = flags

class Shape:
    w: int                 # Max Width
    h: int                 # Max Heigth
    frame: list[FrameRef]  # Frame list

    def __init__(self, w: int, h: int, frame: list[FrameRef]) -> None:
        self.w = w
        self.h = h
        self.frame = frame

    def get_data(self, sort: bool) -> list[Frame]:
        ref: FrameRef
        lst: list[Frame] = []
        for ref in self.frame:
            try:
                # why there is no .find() method?
                lst.index(ref.frm)
            except:
                lst.append(ref.frm)
        if sort:
            lst.sort(key=lambda frm: frm.offset)
        return lst


# ============================================================================ #
#                            Crusader Shape Format                             #
# ============================================================================ #

"""
Scanline format for Compression 0:
    scanline = {skip opcode}.         # stop when scanline fully filled
    skip     = byte.                  # fill N bytes with transparency
    len      = byte.                  #
    opcode   = len copy.              #
    copy     = {byte}.                # copy len bytes

Scanline format for Compression 1:
    scanline = {skip opcode}.         # stop when scanline fully filled
    skip     = byte.                  # fill N bytes with transparency
    len      = 7-bits.                #
    mode     = 1-bit.                 # 0 -> copy | 1 -> fill
    opcode   = len mode (copy|fill).  #
    copy     = {byte}.                # copy len bytes
    fill     = byte.                  # fill len bytes with color X
"""

def _decompress(fp: IO[bytes], comp: int, w: int, h: int,
                limit: int) -> bytearray:
    i: int
    j: int
    base: int
    rows: list[int]
    bitmap: bytearray
    size: int
    typ: int

    assert(w >= 0)
    assert(h >= 0)
    if comp != 0 and comp != 1:
        raise ShapeError("unsupported compression method " + str(comp))

    # read table with scanline offsets
    if 4 * h > limit:
        raise ShapeError("compressed data too small")
    rows = [0] * h
    base = fp.tell()
    for j in range(h):
        rows[j] = struct.unpack("<L", fp.read(4))[0]

    # read scanlines
    bitmap = bytearray(w * h)
    bitmap[:] = b"\xFF" * len(bitmap)
    for j in range(h):
        i = 0
        if rows[j] + 4*j >= limit:
            raise ShapeError("compressed data too small")
        fp.seek(base + 4*j + rows[j])
        while i < w:
            # skip / transparency
            if fp.tell() + 1 > base + limit:
                raise ShapeError("compressed data too small")
            i += fp.read(1)[0]
            if i >= w:
                break;

            # read opcode
            if fp.tell() + 1 > base + limit:
                raise ShapeError("compressed data too small")
            typ = 0
            size = fp.read(1)[0]
            if comp == 1:
                typ = size & 1
                size = size >> 1
            if size > w - i:
                raise ShapeError("invalid scanline opcode")

            # execute opcode
            if typ == 0:
                if fp.tell() + size > base + limit:
                    raise ShapeError("compressed data too small")
                bitmap[w*j+i:w*j+i+size] = fp.read(size)
            else:
                if fp.tell() + 1 > base + limit:
                    raise ShapeError("compressed data too small")
                bitmap[w*j+i:w*j+i+size] = fp.read(1)[:] * size
            i += size

        if i != w:
            raise ShapeError("invalid scanline")

    return bitmap

def read_shp(path: str) -> Shape:
    @dataclass
    class Entry:
        offset: int
        flags: int
        size: int

    def find(lst: list[Frame], offset: int, size: int) -> Optional[Frame]:
        frm: Frame
        for frm in lst:
            if frm.offset == offset and frm.size == size:
                return frm
        return None

    a: tuple[int, ...]  # [offset_low, offset_hi, flag, size]
    b: tuple[int, ...]  # [shp_id, frm_id, flags, comp, w, h, x, y]
    e: Entry
    tab: list[Entry] = []
    frm: Optional[Frame]
    lst: list[Frame] = []
    table: list[FrameRef] = []
    bitmap: bytearray
    fp: IO[bytes]
    size: int
    count: int
    i: int
    w: int
    h: int

    with open(path, "rb") as fp:
        # read header
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        fp.seek(0)
        if size < 6:
            raise ShapeError("file too small to be shape file")
        [w, h, count] = struct.unpack("<HHH", fp.read(6))

        # read table
        if size < 6 + 8 * count:
            raise ShapeError("file too small to be shape file")
        for i in range(count):
            a = struct.unpack("<HBBL", fp.read(8))
            e = Entry(a[0] | (a[1] << 16), a[2], a[3])
            if e.offset + e.size > size:
                raise ShapeError("invalid frame data position or size")
            tab.append(e)

        # read frames
        for e in tab:
            frm = find(lst, e.offset, e.size)
            if frm is None:
                fp.seek(e.offset)
                if e.size < 28:
                    raise ShapeError("invalid frame data size")
                b = struct.unpack("<HHLLLLLL", fp.read(28))
                bitmap = _decompress(fp, b[3], b[4], b[5], e.size - 28)
                frm = Frame(
                    shape = b[0],
                    frame = b[1],
                    flags = b[2],
                    comp = b[3],
                    w = b[4],
                    h = b[5],
                    x = b[6],
                    y = b[7],
                    bitmap = bitmap,
                    offset = e.offset,
                    size = e.size
                )
                lst.append(frm)
            table.append(
                FrameRef(
                    frame = frm,
                    flags = e.flags
                )
            )

    return Shape(w, h, table)

def _scan_rep(line: bytes, start: int, limit: int, pat: int) -> int:
    j: int = start
    i: int = start
    n: int = len(line)

    while (i < n) and (i - j < limit) and (line[i] == pat):
        i += 1

    return i - j

def _compress_scanline(comp: int, line: bytes) -> bytearray:
    THRESHOLD: int = 3 # seems good value
    i: int = 0
    j: int = 0
    mode: int = 0
    limit: int = 0
    n: int = len(line)
    z: bytearray = bytearray()
    count: int = 0

    assert(comp == 0 or comp == 1);
    limit = 127
    if comp == 0:
       limit = 255

    while i < n:
        # skip / transparency length
        j = i
        count = _scan_rep(line, i, 255, 255)
        i += count
        z.append(count)

        if i < n:
            # rle / repeated pattern
            j = i
            i += _scan_rep(line, i, limit, line[j])
            mode = 1

            # preffer copy on small pieces (as original compressor does)
            if i - j < THRESHOLD:
                mode = 0

            # rle not possible for Compression 0
            if comp == 0:
                mode = 0

            # copy pattern
            if mode == 0:
                while (i < n) and (i - j < limit) and (line[i] != 255):
                    count = _scan_rep(line, i, limit, line[i])
                    if count > THRESHOLD:
                        break
                    i += count

            # write opcode
            if comp == 0:
                z.append(i - j)
                z += line[j:i]
            else:
                z.append(((i - j) << 1) + mode)
                if mode == 0:
                    z += line[j:i]
                else:
                    z.append(line[j])
    return z

def _compress(comp: int, w: int, h: int, bitmap: bytearray) -> bytearray:
    y: int                        # current scanline
    offset: int = 0               # last offset
    line: bytearray               # last compressed line
    ofs: list[int] = []           # offset in data
    data: bytearray = bytearray() # result data
    tab: bytearray = bytearray()  # result table

    # compress all scanlines
    for y in range(h):
        line = _compress_scanline(comp, bitmap[w*y:w*(y+1)])

        # deduplicate scanline data
        offset = data.find(line)
        if offset == -1:
            offset = len(data)
            data += line

        ofs.append(offset)

    # generate packed offset table
    for y in range(h):
        tab += struct.pack("<L", ofs[y] + (4*h - 4*y))

    return tab + data

def write_shp(path: str, shp: Shape) -> None:
    ofs: int         # offset to frame table
    count: int       # number of frames
    lst: list[Frame] # list of frames
    data: bytearray
    data0: bytearray
    data1: bytearray
    fp: IO[bytes]

    with open(path, "wb") as fp:
        # write header
        count = len(shp.frame)
        fp.write(struct.pack("<HHH", shp.w, shp.h, count))

        # write empty table (fixup later)
        ofs = fp.tell()
        fp.write(b"\0" * 8*count)

        # write frames
        lst = shp.get_data(True)
        for frm in lst:
            frm.offset = fp.tell()

            # compress data
            if frm.comp < 0:
                data0 = _compress(0, frm.w, frm.h, frm.bitmap)
                data1 = _compress(1, frm.w, frm.h, frm.bitmap)
                data = data1
                frm.comp = 1
                if len(data0) <= len(data1):
                    data = data0
                    frm.comp = 0
            else:
                data = _compress(frm.comp, frm.w, frm.h, frm.bitmap)

            # write frame
            fp.write(
                struct.pack("<HHLLLLLL",
                    frm.shape,
                    frm.frame,
                    frm.flags,
                    frm.comp,
                    frm.w,
                    frm.h,
                    frm.x,
                    frm.y
                )
            )
            fp.write(data)
            frm.size = fp.tell() - frm.offset

        # fixup frame table
        fp.seek(ofs)
        for ref in shp.frame:
            fp.write(struct.pack("<L", ref.frm.offset)[:3])
            fp.write(struct.pack("<BL", ref.flags, ref.frm.size))

# ============================================================================ #
#                       Portable Network Graphigs Format                       #
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

def read_png(name: str) -> tuple[bytearray, int, int]:
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

        return bitmap, w, h

# ============================================================================ #
#                             Shape JSON Metadata                              #
# ============================================================================ #

def to_json(shp: Shape) -> tuple[dict[str, Any], list[Frame]]:
    lst: list[Frame] = shp.get_data(True)
    return {
        "format": "Shape",
        "subformat": "Crusader",
        "w": shp.w,
        "h": shp.h,
        #"COUNT": len(shp.table),
        "frame": list(
            map(
                lambda ref: {
                    "id": lst.index(ref.frm),
                    "flags": ref.flags,
                },
                shp.frame
            )
        ),
        "data": list(
            map(
                lambda frm: {
                    #"OFFSET": frm.offset,
                    #"SIZE": frm.size,
                    "shape": frm.shape,
                    "frame": frm.frame,
                    "flags": frm.flags,
                    "method": frm.comp,
                    #"WIDTH": frm.w,
                    #"HEIGHT": frm.h,
                    "x": frm.x,
                    "y": frm.y,
                },
                lst
            )
        )
    }, lst

def write_json(path: str, shp: Shape) -> None:
    fp: IO[str]
    with open(path, "w") as fp:
        json.dump(to_json(shp)[0], fp, indent=2)

def _get_name(path: str, frame: int, maxframe: int, suffix: str) -> str:
    f: str = str(frame).zfill(len(str(maxframe - 1)))
    name: str = str(Path(path).stem) + "_" + f
    return str(Path(name).with_suffix(suffix))

def write_json_png(path: str, shp: Shape, pal: bytearray) -> None:
    i: int
    n: int
    frm: Frame
    lst: list[Frame]
    name: str
    root: Any
    fp: IO[str]

    # generate json structure
    [root, lst] = to_json(shp)

    # write frames
    n = len(lst)
    for i in range(n):
        frm = lst[i]
        name = _get_name(path, i, n, ".png")
        root["data"][i]["file"] = name
        name = str(Path(path).parent / name)
        write_png(name, frm.w, frm.h, frm.x, frm.y, frm.bitmap, pal)

    # write json
    with open(path, "w") as fp:
       json.dump(root, fp, indent=2)

def _chk(value: Any, typ: type, msg: str) -> Any:
    vtyp: type = type(value)
    if vtyp is not typ:
        raise ShapeError(msg
                         + " expect type " + str(typ)
                         + " but found " + str(vtyp))
    return typ(value)

def _idx(arr: Any, key: Union[str, int], typ: type, msg: str) -> Any:
    value: Any
    vtyp: type
    if (type(arr) is not dict) and (type(arr) is not list):
        raise ShapeError(msg + " expect type dict or list but found "
                         + str(type(arr)))
    if key not in arr:
        raise ShapeError(msg
                         + " requires key " + str(key)
                         + " of type " + str(typ))
    value = arr[typing.cast(SupportsIndex, key)]
    vtyp = type(value)
    if vtyp is not typ:
        raise ShapeError(msg
                         + " key " + str(key)
                         + " expect type " + str(typ)
                         + " but found " + str(vtyp))
    return typ(value)

def _int(value: Any, msg: str,
            imin: Union[int, None]=None,
            imax: Union[int, None]=None) -> int:
    v: int = _chk(value, int, msg)
    if ((imin is not None) and (v < imin)) or ((imax is not None) and (v > imax)):
        raise ShapeError(msg + " must be in range ["
                         + (str(imin), "inf")[imin is None]
                         + ".."
                         + (str(imax), "inf")[imax is None]
                         + "]")
    return v

def read_json_png(path: str) -> Shape:

    root: Any
    o: Any
    shp: Shape
    frame: list[FrameRef] = []
    lst: list[Frame] = []
    bitmap: bytearray
    fp: IO[str]
    name: str
    w: int
    h: int
    i: int

    # read json
    with open(path, "r") as fp:
        root = json.load(fp)

    # check
    _chk(root, dict, "root")
    if _idx(root, "format", str, "root") != "Shape":
        raise ShapeError("not JSON Shape metadata")
    if "subformat" in root:
        _chk(root["subformat"], str, "root")
    _idx(root, "frame", list, "root")
    _idx(root, "data", list, "root")

    # read frame data
    i = 0
    for o in root["data"]:
        _chk(o, dict, "data")
        name = _idx(o, "file", str, "data file")
        name = str(Path(path).parent / name)
        [bitmap, w, h] = read_png(name)
        lst.append(
            Frame(
                offset = i,
                size = 0,
                shape = _int(o.get("shape", 0), "shape id", 0, 2**32-1),
                frame = _int(o.get("frame", 0), "frame id", 0, 2**32-1),
                flags = _int(o.get("flags", 0), "frame flags", 0, 2**32-1),
                comp = _int(o.get("method", -1),
                            "frame compression method", -1, 1),
                w = w,
                h = h,
                x = _int(o.get("x", 0), "frame offset x", 0, 2**32-1),
                y = _int(o.get("y", 0), "frame offset y", 0, 2**32-1),
                bitmap = bitmap
            )
        )
        i += 1

    # read frame table
    for o in root["frame"]:
        _chk(o, dict, "frame")
        i = _idx(o, "id", int, "data id")
        if (i < 0) or (i >= len(lst)):
            raise ShapeError("frame data " + str(i) + " not exists in metadata")
        frame.append(
            FrameRef(
                frame = lst[i],
                flags = _int(o.get("flags", 0), "reference flags", 0, 2**8-1),
            )
        )

    # make shape
    return Shape(
        w = _int(root.get("w", 0), "shape width", 0, 2**32-1),
        h = _int(root.get("h", 0), "shape height", 0, 2**32-1),
        frame = frame
    )

# ============================================================================ #
#                                 VGA Palette                                  #
# ============================================================================ #

def read_pal(path: str) -> bytearray:
    i: int
    size: int
    data: bytearray
    fp: IO[bytes]

    # read palette
    with open(path, "rb") as fp:
        fp.seek(0, os.SEEK_END)
        size = fp.tell();
        fp.seek(0)
        if size != 768:
            raise ShapeError("not VGA palette")
        data = bytearray(768)
        data[:] = fp.read(768)

    # check file
    for i in data:
        if i > 63:
            raise ShapeError("not VGA palette")

    # normalize
    for i in range(768):
        data[i] = int(data[i] / 63 * 255);

    return data

# ============================================================================ #
#                                     Main                                     #
# ============================================================================ #

def _help() -> None:
    fp: IO[str] = sys.stderr
    print("Usage: shp2png <src_fmt> <dst_fmt> <src> <dst> [vgapal]", file=fp)
    print("Formats:", file=fp)
    print("    shp      - Crusader Shape Format", file=fp)
    print("    json     - JSON-formatted metadata", file=fp)
    print("    json+png - JSON-formatted metadata + PNG frames", file=fp)
    exit(1)

def _main() -> int:
    smode: str
    dmode: str
    src: str
    dst: str
    pfn: str
    shp: Shape
    pal: bytearray

    if len(sys.argv) <= 4:
        _help()
    smode = sys.argv[1]
    dmode = sys.argv[2]
    src = sys.argv[3]
    dst = sys.argv[4]
    if len(sys.argv) > 5:
        pfn = sys.argv[5]
    elif dmode == "json+png":
        print("error: palette required for json conversion", file=sys.stderr)
        _help()

    if smode not in ["shp", "json", "json+png"]:
        print("error: unknown source mode " + smode, file=sys.stderr)
        _help()
    if dmode not in ["shp", "json", "json+png"]:
        print("error: unknown destination mode " + smode, file=sys.stderr)
        _help()

    try:
        if dmode == "json+png":
            pal = read_pal(pfn)

        pfn = src
        if smode == "shp":
            shp = read_shp(src)
        elif (smode == "json") or (smode == "json+png"):
            shp = read_json_png(src)

        pfn = dst
        if dmode == "shp":
            write_shp(dst, shp)
        elif dmode == "json":
            write_json(dst, shp)
        elif dmode == "json+png":
            write_json_png(dst, shp, pal)

    except (ValueError, OSError, json.JSONDecodeError) as e:
        print("error: " + pfn + ": " + str(e), file=sys.stderr)
        exit(1)

    return 0

if __name__ == "__main__":
    exit(_main())
