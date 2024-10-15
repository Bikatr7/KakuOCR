from PIL import Image
import bitstring

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
        img = record['Image Data']
        yield char, img