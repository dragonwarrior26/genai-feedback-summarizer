from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'GenAI Feedback Summarizer - Final Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 16)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, label, 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_subtitle(self, label):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(2)
        
    def chapter_subsubtitle(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, label, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, text)
        self.ln()

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 11)
        self.cell(5)  # Indent
        self.multi_cell(0, 6, f"- {text}")
        self.ln(1)
        
    def code_block(self, text):
        self.set_font('Courier', '', 9)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 5, text, fill=True)
        self.ln()

def sanitize_text(text):
    replacements = {
        '→': '->',
        '✅': '[OK]',
        '⚠️': '[WARN]',
        '📌': '',
        '🛠️': '',
        '🚀': '',
        '📊': '',
        '📂': '',
        '📜': '',
        '🤖': '',
        '📝': '',
        '🏷️': '',
        '🔍': '',
        '💡': '',
        '🌐': '',
        '🎨': '',
        '🎯': '',
        '🔧': '',
        '📈': '',
        '🌟': '',
        '📚': '',
        '🤝': '',
        '📄': '',
        '🙏': '',
        '📧': '',
        '⭐': '',
        '–': '-',
        '—': '-',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'"
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf(input_file, output_file):
    pdf = PDF()
    pdf.add_page()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_content = []
    
    for line in lines:
        line = line.rstrip()
        line = sanitize_text(line)
        
        # Handle Code Blocks
        if line.startswith('```'):
            if in_code_block:
                # End of code block
                pdf.code_block('\n'.join(code_content))
                code_content = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
            continue
            
        if in_code_block:
            code_content.append(line)
            continue
            
        # Skip empty lines if previous was header
        if not line:
            continue
            
        # Headers
        if line.startswith('# '):
            pdf.add_page() # New page for main title/chapters
            pdf.chapter_title(line[2:])
        elif line.startswith('## '):
            pdf.chapter_title(line[3:])
        elif line.startswith('### '):
            pdf.chapter_subtitle(line[4:])
        elif line.startswith('#### '):
            pdf.chapter_subsubtitle(line[5:])
            
        # Bullet points
        elif line.strip().startswith('- '):
            pdf.bullet_point(line.strip()[2:])
        elif line.strip().startswith('* '):
            pdf.bullet_point(line.strip()[2:])
            
        # Bold text (simple handling)
        elif '**' in line:
            # Very basic bold handling: just remove stars and print
            clean_line = line.replace('**', '')
            pdf.body_text(clean_line)
            
        # Normal text
        else:
            pdf.body_text(line)
            
    pdf.output(output_file)
    print(f"PDF generated: {output_file}")

if __name__ == '__main__':
    create_pdf('docs/FINAL_REPORT.md', 'docs/FINAL_REPORT.pdf')
