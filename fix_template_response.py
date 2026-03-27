"""
Fixes all templates.TemplateResponse() calls from old Starlette format to new format.

OLD (breaks in Starlette 0.27+):
    templates.TemplateResponse("x.html", {"request": request, "key": val})

NEW (works in all versions):
    templates.TemplateResponse(request=request, name="x.html", context={"key": val})

Usage:
    python fix_template_response.py app.py
"""

import re
import sys
import shutil

def fix_template_responses(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Backup original
    shutil.copy(filepath, filepath + ".bak")
    print(f"Backup saved to {filepath}.bak")

    # Pattern: templates.TemplateResponse(\n?    "name.html",\n?    {\n?        "request": request,\n?        ...
    # We'll do a token-based replacement by finding each call manually

    result = []
    i = 0
    changes = 0

    while i < len(content):
        # Look for templates.TemplateResponse(
        marker = 'templates.TemplateResponse('
        idx = content.find(marker, i)
        if idx == -1:
            result.append(content[i:])
            break

        # Append everything before this match
        result.append(content[i:idx])

        # Find the matching closing paren
        start = idx + len(marker)
        depth = 1
        j = start
        while j < len(content) and depth > 0:
            if content[j] == '(':
                depth += 1
            elif content[j] == ')':
                depth -= 1
            j += 1
        # content[start:j-1] is the arguments
        args_str = content[start:j-1]

        # Check if it's already in new format (has request= as keyword)
        if 'request=request' in args_str or 'name=' in args_str:
            # Already new format, keep as-is
            result.append(marker + args_str + ')')
            i = j
            continue

        # Parse: first arg should be a string (template name)
        # second arg should be a dict starting with "request": request
        stripped = args_str.strip()

        # Find the template name (first quoted string)
        name_match = re.match(r'\s*(["\'])(.+?)\1', stripped)
        if not name_match:
            # Can't parse, keep original
            result.append(marker + args_str + ')')
            i = j
            continue

        template_name = name_match.group(2)
        quote_char = name_match.group(1)

        # After the name, there should be a comma then a dict
        after_name = stripped[name_match.end():].lstrip()
        if not after_name.startswith(','):
            # No second argument, just a name
            result.append(f'templates.TemplateResponse(request=request, name={quote_char}{template_name}{quote_char})')
            i = j
            changes += 1
            continue

        after_comma = after_name[1:].lstrip()

        # The rest is the context dict (possibly with status_code= at the end)
        # Find the main dict boundaries
        if not after_comma.startswith('{'):
            # Unusual format, keep original
            result.append(marker + args_str + ')')
            i = j
            continue

        # Extract the dict
        dict_start = after_comma.index('{')
        dict_depth = 0
        k = dict_start
        while k < len(after_comma):
            if after_comma[k] == '{':
                dict_depth += 1
            elif after_comma[k] == '}':
                dict_depth -= 1
                if dict_depth == 0:
                    break
            k += 1
        
        dict_str = after_comma[dict_start:k+1]
        after_dict = after_comma[k+1:].strip()

        # Remove "request": request from the dict
        # Handle both '"request": request' and '"request":request' with varying whitespace
        cleaned_dict = re.sub(
            r'"request"\s*:\s*request\s*,?\s*',
            '',
            dict_str
        )
        # Also remove trailing comma before closing brace
        cleaned_dict = re.sub(r',\s*}', '\n            }', cleaned_dict)
        # Clean up empty dicts
        cleaned_dict = re.sub(r'{\s*}', '{}', cleaned_dict)

        # Check if there's a status_code kwarg after the dict
        status_code_match = re.search(r',?\s*status_code\s*=\s*(\d+)', after_dict)
        status_code_str = ''
        if status_code_match:
            status_code_str = f',\n            status_code={status_code_match.group(1)}'

        # Build new call
        new_call = (
            f'templates.TemplateResponse(\n'
            f'            request=request,\n'
            f'            name={quote_char}{template_name}{quote_char},\n'
            f'            context={cleaned_dict}'
            f'{status_code_str}\n'
            f'        )'
        )

        result.append(new_call)
        i = j
        changes += 1

    new_content = ''.join(result)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Done! Fixed {changes} TemplateResponse calls in {filepath}")
    print(f"Original backed up to {filepath}.bak")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_template_response.py <path_to_app.py>")
        sys.exit(1)
    fix_template_responses(sys.argv[1])