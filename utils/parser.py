import re
import pandas as pd
from datetime import datetime

# Regex patterns to handle different WhatsApp export formats
# Format 1: DD/MM/YY, HH:MM<U+202F>am/pm
# Example: 02/05/25, 10:22 am - Author Name: Message content
USER_MESSAGE_RE_NBSPACE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*\u202F(?:am|pm|AM|PM))\s*-\s*(.+?):\s*(.*)"
)
SYSTEM_MESSAGE_RE_NBSPACE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*\u202F(?:am|pm|AM|PM))\s*-\s*(.*)"
)

# Format 2: MM/DD/YY, H:MM AM/PM - Author: Message (Common US format)
# Example: 05/02/25, 10:22 AM - Author Name: Message content
USER_MESSAGE_RE_SPACE_AMPN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*-\s*(.+?):\s*(.*)"
)
SYSTEM_MESSAGE_RE_SPACE_AMPN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*-\s*(.*)"
)

# Format 3: DD/MM/YY, HH:MM - Author: Message (24-hour format)
# Example: 23/06/21, 10:30 - John Doe: Hello there
USER_MESSAGE_RE_24H = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2})\s*-\s*(.+?):\s*(.*)"
)
SYSTEM_MESSAGE_RE_24H = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2})\s*-\s*(.*)"
)


REGEX_PAIRS = [
    (USER_MESSAGE_RE_NBSPACE, SYSTEM_MESSAGE_RE_NBSPACE),
    (USER_MESSAGE_RE_SPACE_AMPN, SYSTEM_MESSAGE_RE_SPACE_AMPN),
    (USER_MESSAGE_RE_24H, SYSTEM_MESSAGE_RE_24H),
]

def parse_whatsapp_chat(chat_file_content: str) -> pd.DataFrame:
    """
    Parses a WhatsApp chat export .txt file content.
    Handles various date/time formats and system messages.
    """
    lines = chat_file_content.strip().split('\n')
    parsed_data = []

    current_message_datetime_obj = None
    current_message_author_str = None
    current_message_text_parts = []
    current_message_is_system_flag = False # True if parsed by a SYSTEM_MESSAGE_RE

    def attempt_parse_datetime_str(date_str_val: str):
        if not date_str_val:
            return None

        date_str_val = date_str_val.strip()
        # Pre-normalize am/pm for space-separated AM/PM
        normalized_date_str = date_str_val
        if " am" in date_str_val.lower():
            normalized_date_str = re.sub(r"(?i)\s+am$", " AM", normalized_date_str)
        elif " pm" in date_str_val.lower():
            normalized_date_str = re.sub(r"(?i)\s+pm$", " PM", normalized_date_str)

        # For narrow no-break space, ensure am/pm is lowercase as strptime expects for %p with U+202F
        # The regexes capture 'am' or 'pm' literally for U+202F case
        formats_to_try = [
            # Formats with NARROW NO-BREAK SPACE (U+202F)
            "%d/%m/%y, %I:%M\u202F%p",  # e.g., 02/05/25, 10:22 am
            "%d/%m/%Y, %I:%M\u202F%p", # e.g., 02/05/2025, 10:22 am
            "%m/%d/%y, %I:%M\u202F%p",
            "%m/%d/%Y, %I:%M\u202F%p",

            # Formats with regular space and AM/PM
            "%d/%m/%y, %I:%M %p",    # e.g., 01/07/22, 9:00 AM
            "%d/%m/%Y, %I:%M %p",
            "%m/%d/%y, %I:%M %p",
            "%m/%d/%Y, %I:%M %p",

            # 24-hour formats
            "%d/%m/%y, %H:%M",       # e.g., 23/06/21, 10:30
            "%d/%m/%Y, %H:%M",
            "%m/%d/%y, %H:%M",
            "%m/%d/%Y, %H:%M",
        ]

        test_date_str = date_str_val # Use original for U+202F attempts

        for fmt in formats_to_try:
            current_test_str = normalized_date_str # Default to space-normalized
            if "\u202F" in fmt:
                current_test_str = test_date_str # Use original for U+202F

            try:
                if "\u202F%p" in fmt:
                     # strptime expects 'am' or 'pm' (lowercase) for certain locales/setups with %p
                     # If the regex captured AM/PM, we might need to convert to lowercase for these formats.
                    temp_str = current_test_str.replace("\u202FAM", "\u202Fam").replace("\u202FPM", "\u202Fpm")
                    return datetime.strptime(temp_str, fmt)

                return datetime.strptime(current_test_str, fmt)
            except ValueError:
                # If U+202F format failed, try with opposite case for am/pm just in case
                if "\u202F%p" in fmt:
                    try:
                        temp_str_upper = current_test_str.replace("\u202Fam", "\u202FAM").replace("\u202Fpm", "\u202FPM")
                        return datetime.strptime(temp_str_upper, fmt)
                    except ValueError:
                        continue # Try next format
                continue # Try next format
        # print(f"Warning: Could not parse date: '{date_str_val}' with any format.")
        return None

    def finalize_current_message():
        nonlocal current_message_datetime_obj, current_message_author_str
        nonlocal current_message_text_parts, current_message_is_system_flag

        if current_message_text_parts and current_message_datetime_obj:
            full_message = "\n".join(current_message_text_parts).strip()

            # Strip "<This message was edited>" tag
            edit_tag = "<This message was edited>"
            if full_message.endswith(edit_tag):
                full_message = full_message[:-len(edit_tag)].strip()

            # Determine author: "System" if parsed by system regex or no author found
            author_to_store = "System"
            if not current_message_is_system_flag and current_message_author_str:
                author_to_store = current_message_author_str

            # Determine message type
            message_type = "text"
            lower_full_message = full_message.lower()

            if lower_full_message == "<media omitted>" or \
               lower_full_message == "image omitted" or \
               lower_full_message == "video omitted" or \
               lower_full_message == "sticker omitted" or \
               lower_full_message == "audio omitted" or \
               lower_full_message == "gif omitted":
                message_type = "media"
            elif "(file attached)" in lower_full_message:
                message_type = "file"
                # Example: "Walunj.vcf (file attached)" -> message can be "Walunj.vcf"
            elif full_message == "This message was deleted" or \
                 full_message == "You deleted this message": # Case sensitive as per WA
                message_type = "deleted"

            # Handle system messages that might still have user-like names (e.g., "User X created group")
            # The `current_message_is_system_flag` helps identify these.
            # If a line like "02/05/25, 10:22 am - User X created group" is parsed by SYSTEM_MESSAGE_RE,
            # author_to_store will be "System" and full_message will be "User X created group".
            # is_system_final checks if the author is "System" OR if it was flagged as system
            is_system_final = (author_to_store == "System") or current_message_is_system_flag

            parsed_data.append({
                "datetime": current_message_datetime_obj,
                "author": author_to_store,
                "message": full_message,
                "is_system": is_system_final,
                "message_type": message_type
            })

        # Reset for the next message
        current_message_text_parts = []
        # current_message_datetime_obj, current_message_author_str, current_message_is_system_flag
        # will be overwritten by the next new message line or remain for continuations.

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        matched_new_message = False
        dt_str, author_str, msg_str = None, None, None

        for user_re, system_re in REGEX_PAIRS:
            user_match = user_re.match(line)
            if user_match:
                dt_str, author_str, msg_str = user_match.groups()
                current_message_is_system_flag = False
                matched_new_message = True
                break

            system_match = system_re.match(line)
            if system_match:
                dt_str, msg_str = system_match.groups()
                author_str = None # System messages don't have an explicit author field in this context
                current_message_is_system_flag = True
                matched_new_message = True
                break

        if matched_new_message:
            finalize_current_message() # Finalize any previous message being built

            current_message_datetime_obj = attempt_parse_datetime_str(dt_str)
            if current_message_datetime_obj is None:
                # Failed to parse date from a line that looked like a new message.
                # This could be a malformed line or a multiline message part that resembles a header.
                # If there's an ongoing message, append to it.
                if current_message_text_parts: # Check if there's a previous message context
                    current_message_text_parts.append(line)
                # else:
                    # print(f"Skipping line: Unparseable date and no previous message context: {line}")
                continue # Move to the next line

            current_message_author_str = author_str.strip() if author_str else None
            current_message_text_parts.append(msg_str.strip())

        elif current_message_datetime_obj: # If no new message match, but we have an active message (datetime set)
            # This is a continuation of a multi-line message
            current_message_text_parts.append(line)
        # else:
            # This line is not a start of a new message and there's no active message context.
            # It could be a header/footer or an unparseable line at the beginning.
            # print(f"Skipping unparseable initial line: {line}")

    finalize_current_message() # Finalize the very last message in the file

    if not parsed_data:
        return pd.DataFrame(columns=["datetime", "author", "message", "is_system", "message_type"])

    df = pd.DataFrame(parsed_data)
    if 'datetime' in df.columns and not df['datetime'].isnull().all():
         df['datetime'] = pd.to_datetime(df['datetime']) # Ensure it's datetime type
         df = df.sort_values(by="datetime").reset_index(drop=True)
    
    
    return df
