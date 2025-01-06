class IMDPdataset(Dataset):
    def __init__(self,text,label,tokenizer):
        self.text=text
        self.label=label
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.text)

    @staticmethod
    def cleaned_text(text):
        text=re.sub(r'<.*?>',"",text)
        text=re.sub(r'[^a-zA-z\s]',"",text)
        text=text.lower()
        text=re.sub(r'\s+'," ",text).strip()
        return text
    
    def __getitem__(self,idx):
        cleaned=self.cleaned_text(self.text[idx])
        x=tokenizer(cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
        return{"input_ids":x['input_ids'].squeeze(0),
               "attention_mask":x['attention_mask'].squeeze(0),
               "label":self.label[idx]}