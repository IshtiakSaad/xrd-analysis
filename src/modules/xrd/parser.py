
import io
import pandas as pd
import re

class XRDParser:
    """
    State-of-the-art parser for XRD machine exports.
    Handles messy headers, metadata, and bracketed legends.
    """
    
    @staticmethod
    def parse(file_content: bytes | str) -> pd.DataFrame:
        if isinstance(file_content, bytes):
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                text = file_content.decode('latin-1')
        else:
            text = file_content

        lines = text.splitlines()
        
        # 1. Heuristic: Look for data block using regex
        # This handles "<2Theta> < I >" and variable spacing/brackets
        header_pattern = re.compile(r"<\s*2?theta\s*>.*<\s*i\s*>", re.I)
        
        data_start_idx = -1
        for i, line in enumerate(lines):
            if header_pattern.search(line) or "2theta" in line.lower() and "intensity" in line.lower():
                data_start_idx = i + 1
                break
        
        # 2. Heuristic: Numeric block fallback
        if data_start_idx == -1:
            for i, line in enumerate(lines):
                if not line.strip(): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        # Test if first two parts are numeric after stripping junk
                        p1 = re.sub(r'[^\d.+eE-]', '', parts[0])
                        p2 = re.sub(r'[^\d.+eE-]', '', parts[1])
                        float(p1); float(p2)
                        
                        # Peek ahead to confirm a block
                        confirmations = 0
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if not lines[j].strip(): continue
                            next_parts = lines[j].strip().split()
                            if len(next_parts) >= 2:
                                np1 = re.sub(r'[^\d.+eE-]', '', next_parts[0])
                                np2 = re.sub(r'[^\d.+eE-]', '', next_parts[1])
                                try:
                                    float(np1); float(np2)
                                    confirmations += 1
                                except ValueError: break
                        
                        if confirmations >= 2:
                            data_start_idx = i
                            break
                    except (ValueError, IndexError):
                        continue

        if data_start_idx == -1:
            raise ValueError("Could not find start of numeric XRD data block.")


        # 3. Read and Clean
        data_str = "\n".join(lines[data_start_idx:])
        # Use comma OR whitespace as separator
        df = pd.read_csv(io.StringIO(data_str), sep=r'[,\s\t]+', header=None, engine='python')
        
        # Take first two columns regardless of total count
        df = df.iloc[:, :2]
        df.columns = ["two_theta", "intensity"]
        
        # Numeric conversion with safety: strip anything that isn't a number/dot/sign/e
        def clean_numeric(val):
            if pd.isna(val): return val
            s = str(val)
            # Remove everything except digits, dots, e, E, +, -
            cleaned = re.sub(r'[^0-9.eE+-]', '', s)
            return cleaned

        for col in df.columns:
            df[col] = df[col].apply(clean_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna().reset_index(drop=True)
        return df
