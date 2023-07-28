use serde_json;
use serde_json::ser::Formatter;

use std::io::Error;
use std::io::Write as IWrite;

use core::result;


macro_rules! tri {
    ($e:expr $(,)?) => {
        match $e {
            core::result::Result::Ok(val) => val,
            core::result::Result::Err(err) => return core::result::Result::Err(err),
        }
    };
}

pub type Result<T> = result::Result<T, Error>;

pub trait Write {
    fn write(&mut self, buf: &[u8]) -> Result<usize>;

    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        let result = self.write(buf);
        debug_assert!(result.is_ok());
        debug_assert_eq!(result.unwrap_or(0), buf.len());
        Ok(())
    }

    fn flush(&mut self) -> Result<()>;
}


/// This structure pretty prints a JSON value to make it human readable.
#[derive(Clone, Debug)]
pub struct PrettyFormatter2<'a> {
    current_indent: usize,
    has_value: bool,
    nests: usize,
    newline: bool,
    indent: &'a [u8],
}

impl<'a> PrettyFormatter2<'a> {
    /// Construct a pretty printer formatter that defaults to using two spaces for indentation.
    pub fn new() -> Self {
        PrettyFormatter2 {
            current_indent: 0,
            has_value: false,
            nests: 0,
            newline: false,
            indent: b"  ",
        }
    }
}

impl<'a> Default for PrettyFormatter2<'a> {
    fn default() -> Self {
        PrettyFormatter2::new()
    }
}

impl<'a> Formatter for PrettyFormatter2<'a> {
    #[inline]
    fn begin_array<W>(&mut self, writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        self.current_indent += 1;
        self.nests += 1;
        self.has_value = false;
        writer.write_all(b"[")
    }

    #[inline]
    fn end_array<W>(&mut self, writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        self.current_indent -= 1;
        self.nests -= 1;
        if self.nests > 1 {
            self.newline = true;
        }
        else {
            self.newline = false;
        }
        writer.write_all(b"]")
    }

    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        if first {
            Ok(())
        } else {
            if self.newline {
                self.newline = false;
                tri!(writer.write_all(b",\n"));
                indent(writer, self.current_indent, self.indent)
            }
            else {
                writer.write_all(b",")
            }
            // writer.write_all(b",")
        }
    }

    #[inline]
    fn end_array_value<W>(&mut self, _writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        self.has_value = true;
        Ok(())
    }

    #[inline]
    fn begin_object<W>(&mut self, writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        self.current_indent += 1;
        self.has_value = false;
        writer.write_all(b"{")
    }

    #[inline]
    fn end_object<W>(&mut self, writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        self.current_indent -= 1;

        if self.has_value {
            tri!(writer.write_all(b"\n"));
            tri!(indent(writer, self.current_indent, self.indent));
        }

        writer.write_all(b"}")
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        tri!(writer.write_all(if first { b"\n" } else { b",\n" }));
        indent(writer, self.current_indent, self.indent)
    }

    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        writer.write_all(b": ")
    }

    #[inline]
    fn end_object_value<W>(&mut self, _writer: &mut W) -> Result<()>
    where
        W: ?Sized + IWrite,
    {
        self.has_value = true;
        Ok(())
    }
}

fn indent<W>(wr: &mut W, n: usize, s: &[u8]) -> Result<()>
where
    W: ?Sized + IWrite,
{
    for _ in 0..n {
        tri!(wr.write_all(s));
    }

    Ok(())
}
