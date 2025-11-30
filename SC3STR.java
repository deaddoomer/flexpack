import java.lang.*;
import java.io.*;

final class SC3STR {

  private static final String charsetName = "CP866";

  private static String unpack (byte[] data, int[] mtab, int otab[], byte[] strtab) throws IOException
  {
    int bits = 0;
    int buf = 0;
    int i = 0;
    int strlen = 0;
    boolean copy = false;
    String str = "";

    while (i != data.length) {
      int ch = mtab.length - 2;
      do {
        if (bits == 0) {
          buf = data[i++] & 0xff;
          bits = 8;
        }
        ch = mtab[ch | (buf & 1)];
        buf >>= 1;
        bits--;
      } while (ch < 0x8000);
      ch = (-ch - 1) & 0xff;

      if (copy) {
        if (ch < otab.length) {
          for (strlen = 0; strtab[otab[ch] + strlen] != 0; strlen++);
          str += new String(strtab, otab[ch], strlen, charsetName);
        }
        copy = false;
      } else if (ch == 26 && otab != null && otab.length != 0) {
        copy = true;
      } else if (i != data.length || ch != 0) {
        str += new String(new byte[] {(byte)ch}, charsetName);
      }
    }

    return str;
  }

  private static String[][] loadText (LittleEndianDataInput in) throws IOException
  {
    int segments = in.readUnsignedShort(); // up-to 256
    int[] segmentSize = new int[segments];
    int[] segmentStrings = new int[segments];
    for (int i = 0; i < segments; i++) {
      segmentStrings[i] = in.readUnsignedShort();
      segmentSize[i] = in.readInt();
    }

    int[][] segmentPackedSize = new int[segments][];
    for (int i = 0; i < segments; i++) {
      int n = segmentStrings[i];
      segmentPackedSize[i] = new int[n];
      for (int j = 0; j < n; j++) {
        segmentPackedSize[i][j] = in.readUnsignedShort();
      }
    }

    int[] mtab = new int[in.readUnsignedShort()];
    for (int i = 0; i < mtab.length; i++) {
      mtab[i] = in.readUnsignedShort();
    }

    int[] otab = new int[in.readUnsignedShort()]; // up-to 256
    byte[] strtab = new byte[0];
    if (otab.length != 0) {
      for (int i = 0; i < otab.length; i++) {
        otab[i] = in.readUnsignedShort();
      }
      strtab = new byte[in.readUnsignedShort()];
      for (int i = 0; i < strtab.length; i++) {
        strtab[i] = in.readByte();
      }
    }

    int segmentPos = in.position();
    String[][] text = new String[segments][];
    for (int i = 0; i < text.length; i++) {
      text[i] = new String[segmentStrings[i]];

      in.setPosition(segmentPos);
      for (int j = 0; j < text[i].length; j++) {
        byte[] buf = new byte[segmentPackedSize[i][j]]; // up-to 3072
        in.read(buf);

        if (mtab.length > 1) {
          text[i][j] = unpack(buf, mtab, otab, strtab);
        } else {
          // uncompressed strings in russian version
          text[i][j] = new String(buf, 0, buf.length - 1, charsetName);
        }
      }

      segmentPos += segmentSize[i];
    }

    return text;
  }

  private static void saveText (LittleEndianDataOutput out, String[][] text) throws IOException
  {
    // writes uncompressed variant
    out.writeShort(text.length);                 // num segments
    for (int i = 0; i < text.length; i++) {
      int size = 0;
      for (int j = 0; j < text[i].length; j++) {
        size += text[i][j].length() + 1;
      }
      out.writeShort(text[i].length);            // num strings in segment
      out.writeInt(size);                        // segment size
    }

    for (int i = 0; i < text.length; i++) {
      for (int j = 0; j < text[i].length; j++) {
        out.writeShort(text[i][j].length() + 1); // string length + null terminator
      }
    }

    out.writeShort(1);                           // mtab size
    out.writeShort(0);                           // mtab[0]
    out.writeShort(1);                           // otab size
    out.writeShort(1);                           // otab[0]
    out.writeShort(2);                           // strtab size
    out.writeByte(33);                           // strtab[0]
    out.writeByte(0);                            // strtab[1]

    for (int i = 0; i < text.length; i++) {
      for (int j = 0; j < text[i].length; j++) {
        out.write(text[i][j].getBytes(charsetName));
        out.writeByte(0);
      }
    }
  }

  public static void main (String[] args) throws FileNotFoundException, IOException
  {
    if (args.length != 2) {
      System.out.println("usage: SC3STR <input.DAT> <output.DAT>");
      System.exit(1);
    }  

    LittleEndianDataInput in = new LittleEndianDataInput(new FileInputStream(args[0]));
    String[][] text = loadText(in);
    in.close();

    LittleEndianDataOutput out = new LittleEndianDataOutput(new FileOutputStream(args[1]));
    saveText(out, text);
    out.close();
  }
}
