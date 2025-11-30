import java.io.*;

final class LittleEndianDataInput {
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

  public int position ()
  {
    return cpos;
  }

  public void setPosition (int pos) throws IOException
  {
    skipBytes(pos - cpos);
  }
}
