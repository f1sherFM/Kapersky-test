import argparse
from pyfiglet import Figlet
from termcolor import colored
import sys
from app import main as app_main

def print_banner():
    f = Figlet(font='slant')
    print(colored(f.renderText('TEXT ANALYZER'), 'cyan'))
    print(colored("="*60, 'light_blue'))
    print(colored("Professional Content Comparison Tool v1.0", 'yellow'))
    print(colored("="*60, 'light_blue'))
    print()

def parse_args():
    parser = argparse.ArgumentParser(description='Compare original web content with extracted text')
    parser.add_argument('-i', '--input', default='Test_check.csv', 
                        help='Input CSV file path')
    parser.add_argument('-o', '--output', default='results',
                        help='Output directory for reports')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--excel', action='store_true',
                        help='Generate Excel report')
    parser.add_argument('--parallel', type=int, default=4, 
                   help='Number of parallel workers')
    return parser.parse_args()

def main():
    print_banner()
    args = parse_args()
    
    try:
        # Здесь можно добавить проверку входных параметров
        print(colored("[*] Starting analysis...", 'green'))
        
        # Вызов основной логики
        app_main()
        
        print(colored("\n[+] Analysis completed successfully!", 'green'))
        print(colored(f"Reports saved to: {args.output}", 'blue'))
        
    except Exception as e:
        print(colored(f"\n[!] Error: {str(e)}", 'red'))
        sys.exit(1)

if __name__ == "__main__":
    main()