import java.lang.*;
import java.io.*;

final class QTool {

  private static final boolean debug = false;

  private static final class LittleEndianDataOutput {
    private OutputStream stream;
    private int written;

    public LittleEndianDataOutput (OutputStream stream)
    {
      assert(stream != null);
      this.stream = stream;
      this.written = 0;
    }

    public void close () throws IOException
    {
      stream.close();
    }

    public void write (byte[] data) throws IOException
    {
      stream.write(data);
      written += data.length;
    }

    public void writeByte (int value) throws IOException
    {
      stream.write(value);
      written++;
    }

    public void writeShort (int value) throws IOException
    {
      write(new byte[] {(byte)value, (byte)(value >> 8)});
    }

    public void writeInt (int value) throws IOException
    {
      write(new byte[] {(byte)value, (byte)(value >> 8), (byte)(value >> 16), (byte)(value >> 24)});
    }
  }

  private static final class LittleEndianDataInput {
    private InputStream stream;
    private int cpos;

    public LittleEndianDataInput (InputStream stream)
    {
      assert(stream != null);
      this.stream = stream;
      this.cpos = 0;
    }

    public void close () throws IOException
    {
      stream.close();
    }

    public void read (byte[] data) throws IOException
    {
      int i = 0;
      int n = data.length;
      while (n != 0) {
        int r = stream.read(data, i, n);
        if (r == -1) {
          throw new EOFException();
        } else {
          i += r;
          n -= r;
          cpos += r;
        }
      }
    }

    public byte readByte () throws IOException
    {
      int x = stream.read();
      if (x == -1) throw new EOFException();
      cpos += 1;
      return (byte)x;
    }

    public int readUnsignedByte () throws IOException
    {
      return readByte() & 0xff;
    }

    public int readShort () throws IOException
    {
      return readUnsignedByte() | (readByte() << 8);
    }

    public int readUnsignedShort () throws IOException
    {
      return readUnsignedByte() | (readUnsignedByte() << 8);
    }

    public int readInt () throws IOException
    {
      return readUnsignedByte() | (readUnsignedByte() << 8) | (readUnsignedByte() << 16) | (readByte() << 24);
    }

    public void skipBytes (int n) throws IOException
    {
      if (stream.skip(n) != n)
        throw new EOFException();
      cpos += n;
    }

    public int pos ()
    {
      return cpos;
    }
  }

  private static final class RiffWaveInputStream {
    private LittleEndianDataInput stream;
    private int dataSize;
    private int dataStart;

    public int format;
    public int channels;
    public int rate;
    public int byterate;
    public int alignment;
    public int bits;

    public RiffWaveInputStream (InputStream in) throws IOException, Exception
    {
      stream = new LittleEndianDataInput(in);
      if (stream.readInt() != 0x46464952) {
        throw new Exception("expected RIFF header");
      }
      int riffSize = stream.readInt();
      //if (riffSize > stream.size() - 8) {
      //  throw new Exception("truncated RIFF");
      //}

      if (stream.readInt() != 0x45564157) {
        throw new Exception("expected WAVE header");
      }

      if (stream.readInt() != 0x20746D66) {
        throw new Exception("expected fmt header");
      }
      int fmtSize = stream.readInt();
      if (fmtSize < 16 || stream.pos() + fmtSize > 8 + riffSize) {
        throw new Exception("invalid fmt size");
      }
      this.format    = stream.readUnsignedShort();
      this.channels  = stream.readUnsignedShort();
      this.rate      = stream.readInt();
      this.byterate  = stream.readInt();
      this.alignment = stream.readUnsignedShort();
      this.bits      = stream.readUnsignedShort();
      if (this.byterate != this.rate * this.channels * this.bits / 8) {
        throw new Exception("invalid byterate field");
      }
      if (this.alignment != this.channels * this.bits / 8) {
        throw new Exception("invalid alignment field");
      }
      stream.skipBytes(fmtSize - 16);

      if (stream.readInt() != 0x61746164) {
        throw new Exception("expected data header");
      }
      this.dataSize = stream.readInt();
      if (stream.pos() + dataSize > 8 + riffSize) {
        throw new Exception("invalid data size");
      }
      this.dataStart = stream.pos();
    }

    public void close () throws IOException
    {
      stream.close();
    }

    public int available ()
    {
      return dataSize - (stream.pos() - dataStart);
    }

    public void read (byte[] buf) throws IOException
    {
      if (available() < buf.length) {
        throw new EOFException();
      }
      stream.read(buf);
    }
  }

  private static final void writeRawSegment (int id, int op, int pos, byte[] data) throws IOException, FileNotFoundException
  {
    String fname = "seg" + String.format("%08d", id) + "_op" + op + "_at_" + pos + ".bin";
    FileOutputStream out = new FileOutputStream(fname);
    out.write(data);
    out.close();
  }

  private static final void exportWAV (LittleEndianDataInput in, String outName) throws IOException, Exception
  {
    if (in.readUnsignedShort() != 0x6839) { // magic value
      System.out.println("not Q file container");
      System.exit(1);
    }

    int major   = in.readUnsignedByte();  // - unknown, not used
    int minor   = in.readUnsignedByte();  // - unknown, not used
    int width   = in.readUnsignedShort(); // frame width?
    int height  = in.readUnsignedShort(); // frame height?
    int scale_x = in.readUnsignedByte();  // frame scale x?
    int scale_y = in.readUnsignedByte();  // frame scale y?
    int u0      = in.readUnsignedShort(); // - unknown, not used
    int vidpos  = in.readInt();           // first video stream segment?
    int fps     = in.readUnsignedByte();  // frames per second?
    int u1      = in.readUnsignedByte();  // - unknown, not used
    int size    = in.readUnsignedShort(); // sector size?
    int u2      = in.readUnsignedShort(); // - unknown, not used
    int sndlen  = in.readUnsignedShort(); // sound data fragment size?
    int u3      = in.readUnsignedShort(); // - unknown, not used
    boolean ext = sndlen != 0 && size == 0x8000;
    fps         = fps == 0 ? 5 : fps;

    int segment = 0;
    boolean finish = false;
    FileOutputStream out = null;
    do {
      int pos = in.pos();
      int op = in.readShort();

      if (debug) {
        System.out.println("POS: " + pos);
        System.out.println("OP:  " + op);
      }

      switch (op) {
        case -1:
          finish = true;
          break;
        case 0:    // sound data fragment
        case 1:    // set palette
        case 2:    // ?
        //case 3:  // ? (not tested)
        //case 4:  // ? (not tested)
        case 5:    // ?
        //case 6:  // ? (not tested)
        //case 7:  // ? (not tested)
        case 8:    // sound header
        case 9:    // ?
        case 10:   // align buffer
        case 11:   // ?
        //case 12: // ? (not tested)
          int len = op == 10 ? 0x8000 - in.pos() % 0x8000 : in.readInt();

          if (debug) {
            System.out.println("LEN: " + len);
          }

          byte[] data = new byte[len];
          in.read(data);

          if (op == 8) {
            if (out != null) {
              throw new Exception("multiple wav headers in file");
            }
            out = new FileOutputStream(outName);
            out.write(data);
          } else if (op == 0) {
            if (out == null) {
              throw new Exception("no headers");
            }
            out.write(data);
          } else if (op == 10 && !ext) {
            throw new Exception("aling command not implemented for unaligned Q variant");
          } else {
            // writeRawSegment(segment, op, pos, data);
          }
          break;
        default:
          throw new Exception("unknown segment type " + op + " at " + pos);
      }

      if (debug) {
        System.out.println();
      }

      segment++;
    } while (!finish);

    if (debug) {
      System.out.print("segments: ");
      System.out.println(segment);
    }

    if (out == null) {
      System.out.println("no sound stream");
    } else {
      out.close();
    }
  }

  private static final void importWAV (LittleEndianDataInput in, RiffWaveInputStream riff, LittleEndianDataOutput out) throws IOException, Exception
  {
    if (in.readUnsignedShort() != 0x6839) { // magic value
      System.out.println("not Q file container");
      System.exit(1);
    }

    int major   = in.readUnsignedByte();  // - unknown, not used
    int minor   = in.readUnsignedByte();  // - unknown, not used
    int width   = in.readUnsignedShort(); // frame width?
    int height  = in.readUnsignedShort(); // frame height?
    int scale_x = in.readUnsignedByte();  // frame scale x?
    int scale_y = in.readUnsignedByte();  // frame scale y?
    int u0      = in.readUnsignedShort(); // - unknown, not used
    int vidpos  = in.readInt();           // first video stream segment?
    int fps     = in.readUnsignedByte();  // frames per second?
    int u1      = in.readUnsignedByte();  // - unknown, not used
    int size    = in.readUnsignedShort(); // sector size?
    int u2      = in.readUnsignedShort(); // - unknown, not used
    int sndlen  = in.readUnsignedShort(); // sound data fragment size?
    int u3      = in.readUnsignedShort(); // - unknown, not used

    out.writeShort(0x6839);
    out.writeByte(major);
    out.writeByte(minor);
    out.writeShort(width);
    out.writeShort(height);
    out.writeByte(scale_x);
    out.writeByte(scale_y);
    out.writeShort(u0);
    out.writeInt(vidpos);
    out.writeByte(fps);
    out.writeByte(u1);
    out.writeShort(size);
    out.writeShort(u2);
    out.writeShort(sndlen);
    out.writeShort(u3);

    fps = fps == 0 ? 5 : fps;
    boolean ext = sndlen != 0 && size == 0x8000;

    boolean finish = false;
    boolean wavWritten = false;
    do {
      int pos = in.pos();
      int op = in.readShort();
      out.writeShort(op);
      switch (op) {
        case -1:   // end of stream
        case 0:    // sound data fragment
        case 1:    // set palette
        case 2:    // ?
        //case 3:  // ? (not tested)
        //case 4:  // ? (not tested)
        case 5:    // ?
        //case 6:  // ? (not tested)
        //case 7:  // ? (not tested)
        case 8:    // sound header
        case 9:    // ?
        case 10:   // align buffer
        case 11:   // ?
        //case 12: // ? (not tested)
          int len;
          if (op == 10) {
            if (!ext) {
              throw new Exception("alingn command not implemented for unaligned Q variant");
            }
            len = 0x8000 - in.pos() % 0x8000;
          } else {
            len = in.readInt();
            out.writeInt(len);
          }

          if (op == 8) {
            if (wavWritten) {
              throw new Exception("multiple wav headers in file");
            }
            if (len != 44) {
              throw new Exception("expected 44 byte RIFF header in Q file (found " + len + ")");
            }
            if (riff.format != 1 || riff.bits != 16) {
              throw new Exception("expected S16 PCM format");
            }
            in.skipBytes(len);
            out.writeInt(0x46464952);                // "RIFF"
            out.writeInt(44 - 8 + riff.available()); // RIFF size
            out.writeInt(0x45564157);                // "WAVE"
            out.writeInt(0x20746D66);                // "fmt"
            out.writeInt(16);                        // fmt size
            out.writeShort(riff.format);
            out.writeShort(riff.channels);
            out.writeInt(riff.rate);
            out.writeInt(riff.byterate);
            out.writeShort(riff.alignment);
            out.writeShort(riff.bits);
            out.writeInt(0x61746164);                // "data"
            out.writeInt(riff.available());          // data size
            wavWritten = true;
          } else if (op == 0) {
            if (wavWritten == false) {
              throw new Exception("no headers");
            }
            int rifflen = riff.available();
            int maxlen = Math.min(len, rifflen);
            byte[] data = new byte[maxlen];
            in.skipBytes(len);
            riff.read(data);
            out.write(data);
            if (maxlen < len) {
              out.write(new byte[len - maxlen]);
            }
          } else {
            byte[] data = new byte[len];
            in.read(data);
            out.write(data);
          }

          finish = op == -1;
          break;
        default:
          throw new Exception("unknown segment type " + op + " at " + pos);
      }
    } while (!finish);

    if (!wavWritten) {
      System.out.println("warning: no audio stream in Q file, nothing to do");
    } else if (riff.available() > 0) {
      System.out.println("error: audio stream too long to fit into Q file");
      System.exit(1);
    }
  }

  public static final void main (String[] arg) throws IOException, Exception
  {
    RiffWaveInputStream riff;
    LittleEndianDataInput in;
    LittleEndianDataOutput out;

    if (arg.length < 1) {
      help();
    }

    try {
      switch (arg[0]) {
        case "import":
          if (arg.length < 4) {
            help();
          }
          in = new LittleEndianDataInput(new FileInputStream(arg[1]));
          riff = new RiffWaveInputStream(new FileInputStream(arg[2]));
          out = new LittleEndianDataOutput(new FileOutputStream(arg[3]));
          importWAV(in, riff, out);
          break;
        case "export":
          if (arg.length < 3) {
            help();
          }
          in = new LittleEndianDataInput(new FileInputStream(arg[1]));
          exportWAV(in, arg[2]);
          in.close();
          break;
        default:
          help();
      }
    } catch (FileNotFoundException e) {
      System.out.println("file not found");
      System.exit(1);
    }
  }

  private static final void help ()
  {
    System.out.println("usage: QTool export <input.Q> <output.WAV>");
    System.out.println("       QTool import <input.Q> <input.WAV> <output.Q>");
    System.exit(1);
  }
}
