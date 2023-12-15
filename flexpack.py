#!/usr/bin/env python3

"""
Simple Ultima/Crusader FLeX archive (un)packer.

Used format description:
    struct {
        char comment[80]; // 0x00 // typically filled with 0x1A
        u16  signature;   // 0x50 // 0x1A1A
        u16  ?;           // 0x52 // Unknown, typically 0
        u32  N;           // 0x54 // N files in archive
        u32  ?;           // 0x58 // Unknown, typically 1
        u32  size;        // 0x5C // Archive size or zero
        u8   ?[32]        // 0x60 // Unknown, typically filled with zero
        struct {
            u32 offset;   // 0x80 + 8*i + 0 // Data offset
            u32 size;     // 0x80 + 8*i + 4 // Data size
        } fat[N];         // 0x80           // File Table
        u8 data[];        // Raw Data
    };
"""

import os
import sys
import struct

from pathlib import Path

__author__ = "DeaDDooMER"
__license__ = "MIT"
__version__ = "1.0.1"

def _unpack(path: str, dst: str, show: bool, create: bool) -> None:
    h: bytes
    g: int
    v: int
    n: int
    f: int
    s: int
    u: bytes
    i: int
    fsize: int
    fat: bytes
    offset: int
    size: int
    pad: int
    data: bytes
    fn: str
    with open(path, "rb") as fp:
        fp.seek(0, os.SEEK_END)
        fsize = fp.tell()
        if fsize < 128:
           print("error: file too small to be FLX archive", file=sys.stderr)
           exit(1)
        fp.seek(0, os.SEEK_SET)
        [h, g, v, n, f, s, u] = struct.unpack("<80sHHLLL32s", fp.read(128))
        if g != 0x1A1A:
            print("error: not FLX archive", file=sys.stderr)
            exit(1)
        pad = len(str(n - 1))
        for i in range(80):
            if h[i] != 0x1A:
                print("warning: non 0x1A value at offset " + str(i),
                      file=sys.stderr)
        if v != 0:
            print("warning: value " + str(v) + " (2 bytes) != 0 at offset 82",
                  file=sys.stderr)
        if f != 1:
            print("warning: value " + str(f) + " (4 bytes) != 1 at offset 88",
                  file=sys.stderr)
        if s != 0:
            print("warning: value " + str(s) + " (4 bytes) != " + str(0)
                  + " at offset 92 (archive size field)", file=sys.stderr)
        for i in range(31):
            if u[i] != 0:
                print("warning: value " + str(u[i])
                      + " (1 byte) != 0 at offset "
                      + str(96 + i), file=sys.stderr)
        if fsize < 128 + 8 * n:
           print("error: file too small to fit FAT", file=sys.stderr)
           exit(1)
        fat = fp.read(8 * n)
        for i in range(n):
            [eoffset, esize] = struct.unpack_from("<LL", fat, i * 8)
            if eoffset + esize > fsize:
                print("error: corrupted file: entry " + str(i)
                      + " not in file (offset " + str(eoffset)
                      + " / size " + str(esize)
                      + ")", file=sys.stderr)
                exit(1)
            if show:
                print(str(i) + "/" + str(n - 1)
                      + " (offset " + str(eoffset)
                      + " / size " + str(esize)
                      + ")")
            if create:
                fn = str(Path(dst) / f"{i:0{pad}d}.FXO")
                try:
                    with open(fn, "wb") as fd:
                        fp.seek(eoffset)
                        fd.write(fp.read(esize))
                except OSError:
                    print("error: failed to create file " + fn,
                          file=sys.stderr)
                    exit(1)

def _pack(path: str, files: list[str]) -> None:
    i: int
    n: int = len(files)
    ioffset: int
    data: bytes
    fn: str
    with open(path, "wb") as fd:
        for i in range(82):
             fd.write(b"\x1A")
        fd.write(struct.pack("<HLLL", 0, n, 1, 0))
        for i in range(32):
             fd.write(b"\x00")
        for i in range(n):
            fd.write(struct.pack("<LL", 0, 0))
        ioffset = fd.tell()
        for i in range(n):
            fn = files[i]
            try:
                with open(fn, "rb") as fp:
                    data = fp.read()
            except OSError:
                print("error: failed to open file " + fn, file=sys.stderr)
                exit(1)
            if len(data) != 0:
                # copy data
                fd.write(data)
                # update FAT
                fd.seek(128 + 8 * i)
                fd.write(struct.pack("<LL", ioffset, len(data)))
                # next
                ioffset += len(data)
                fd.seek(ioffset)

def _help(err: bool) -> None:
    print("Usage: flexpack list <file.flx>", file=sys.stderr)
    print("       flexpack extract <file.flx> [dir]", file=sys.stderr)
    print("       flexpack create <file.flx> [file...]", file=sys.stderr)
    print("       flexpack help", file=sys.stderr)
    if err:
        exit(1)
    else:
        exit(0)

def _main() -> int:
    mode: str
    arc: str
    dst: str
    lst: list[str]
    n: int

    if len(sys.argv) < 2:
        _help(True)
    mode = sys.argv[1]

    try:
        if mode in ["list", "extract"]:
            if len(sys.argv) < 3:
                _help(True)
            arc = sys.argv[2]
            if len(sys.argv) > 3:
                dst = sys.argv[3]
            else:
                dst = ""
            _unpack(arc, dst, mode == "list", mode == "extract")
        elif mode == "create":
            if len(sys.argv) < 3:
                _help(True)
            arc = sys.argv[2]
            lst = sys.argv[3:]
            _pack(arc, lst)
        else:
             _help(mode != "help")
    except OSError:
        print("error: failed to " + mode + " archive " + arc, file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    exit(_main())
