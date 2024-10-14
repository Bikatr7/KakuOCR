from PIL import Image
import os
import logging
import bitstring
from collections import Counter

## Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETL8B_Record:
    def __init__(self):
        self.octets_per_record = 512
        self.fields = [
            "Serial Sheet Number", "JIS Kanji Code", "JIS Typical Reading", "Image Data"
        ]
        self.bitstring = 'uint:16,hex:16,bytes:4,bytes:504'
        self.converter = {
            'JIS Typical Reading': lambda x: x.decode('ascii'),
            'Image Data': lambda x: Image.frombytes('1', (64, 63), x, 'raw')
        }

    def read(self, bs):
        r = bs.readlist(self.bitstring)
        record = dict(zip(self.fields, r))
        return {
            k: (self.converter[k](v) if k in self.converter else v)
            for k, v in record.items()
        }

    def get_char(self, record):
        char = bytes.fromhex(
            '1b2442' + record['JIS Kanji Code'] + '1b2842').decode('iso2022_jp')
        return char

def read_record_etl8b(file):
    etl8b_record = ETL8B_Record()
    f = bitstring.ConstBitStream(filename=file)
    f.bytepos = etl8b_record.octets_per_record  ## Skip first record
    while(f.pos < f.length):
        record = etl8b_record.read(f)
        char = etl8b_record.get_char(record)
        yield char, record['Serial Sheet Number'], record['JIS Kanji Code']

## Set working directory dynamically to where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_path = 'ETL8B/ETL8B/ETL8B2C1'

## Read all characters and collect statistics
char_counter = Counter()
total_records = 0

for char, serial, jis_code in read_record_etl8b(file_path):
    char_counter[char] += 1
    total_records += 1

## Print summary
logger.info(f"Total number of records: {total_records}")
logger.info(f"Number of unique characters: {len(char_counter)}")
logger.info("Top 10 most common characters:")
for char, count in char_counter.most_common(10):
    logger.info(f"  '{char}': {count} occurrences")

logger.info("Character distribution:")
logger.info(f"  Most common: '{char_counter.most_common(1)[0][0]}' ({char_counter.most_common(1)[0][1]} occurrences)")
logger.info(f"  Least common: '{char_counter.most_common()[-1][0]}' ({char_counter.most_common()[-1][1]} occurrences)")
