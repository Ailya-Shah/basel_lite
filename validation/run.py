"""One command: validate, print the summary, and write the PDF report."""
from validation.validate import run, _summary
from validation.report import build_pdf

if __name__ == "__main__":
    res = run()
    print(_summary(res))
    print(f"\nReport -> {build_pdf(res)}")
