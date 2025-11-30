import java.io.*;

final class LittleEndianDataOutput {
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
