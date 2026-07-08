"""One command: compute ECL, print the summary, write the PDF."""
from ecl.engine import run, summary
from ecl.report import build_pdf

if __name__ == "__main__":
    res = run()
    print(summary(res))
    print(f"\nReport -> {build_pdf(res)}")
