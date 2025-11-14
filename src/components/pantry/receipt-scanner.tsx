'use client';

import { useState, useRef } from 'react';
import Tesseract from 'tesseract.js';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { FileText, Upload, Loader2 } from 'lucide-react';
import { Input } from '@/components/ui/input';

type ReceiptScannerProps = {
  onItemsExtracted: (items: string[]) => void;
  onClose: () => void;
  isOpen: boolean;
};

export function ReceiptScanner({ onItemsExtracted, onClose, isOpen }: ReceiptScannerProps) {
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processReceipt = async (file: File) => {
    try {
      setError(null);
      setScanning(true);
      setProgress(0);

      const result = await Tesseract.recognize(file, 'eng', {
        logger: (m) => {
          if (m.status === 'recognizing text') {
            setProgress(Math.round(m.progress * 100));
          }
        },
      });

      // Extract items from text (simple parsing)
      const text = result.data.text;
      const lines = text.split('\n').filter(line => line.trim().length > 0);
      
      // Try to extract item names (filtering out prices, dates, etc.)
      const items = lines
        .filter(line => {
          // Skip lines that are mostly numbers or symbols
          const hasLetters = /[a-zA-Z]{3,}/.test(line);
          const isNotPrice = !/^\$?\d+\.?\d*$/.test(line.trim());
          const isNotDate = !/\d{1,2}\/\d{1,2}\/\d{2,4}/.test(line);
          return hasLetters && isNotPrice && isNotDate;
        })
        .map(line => line.trim())
        .filter(line => line.length > 2 && line.length < 50);

      if (items.length === 0) {
        setError('No items found in the receipt. Please try again with a clearer image.');
      } else {
        onItemsExtracted(items);
        onClose();
      }
    } catch (err) {
      setError('Failed to process receipt. Please try again.');
      console.error('OCR error:', err);
    } finally {
      setScanning(false);
      setProgress(0);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      processReceipt(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Scan Receipt</DialogTitle>
          <DialogDescription>
            Upload a photo of your receipt to automatically extract items.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          {error && (
            <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-md">
              {error}
            </div>
          )}
          
          {scanning ? (
            <div className="space-y-4 py-8">
              <div className="flex justify-center">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Processing receipt...</p>
                <p className="text-2xl font-bold mt-2">{progress}%</p>
              </div>
            </div>
          ) : (
            <>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer" onClick={handleUploadClick}>
                <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground">
                  Click to upload a receipt image
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Supports JPG, PNG, and other image formats
                </p>
              </div>
              
              <Input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={onClose}>
                  Cancel
                </Button>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function ReceiptScannerButton({ onItemsExtracted }: { onItemsExtracted: (items: string[]) => void }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button
        variant="outline"
        size="icon"
        onClick={() => setIsOpen(true)}
        title="Scan receipt"
      >
        <FileText className="h-4 w-4" />
      </Button>
      <ReceiptScanner 
        isOpen={isOpen} 
        onClose={() => setIsOpen(false)} 
        onItemsExtracted={(items) => {
          setIsOpen(false);
          onItemsExtracted(items);
        }} 
      />
    </>
  );
}
