'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PlusCircle, Camera, ScanLine } from "lucide-react";
import type { PantryItem } from "@/lib/types";
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';

type AddPantryItemDialogProps = {
  onAddItem: (item: Omit<PantryItem, 'id' | 'riskClass' | 'riskScore'>) => void;
};

export function AddPantryItemDialog({ onAddItem }: AddPantryItemDialogProps) {
  const [open, setOpen] = useState(false);
  const [scanDialogOpen, setScanDialogOpen] = useState(false);
  const [name, setName] = useState('');
  const [quantity, setQuantity] = useState(1);
  const [unit, setUnit] = useState<'kg' | 'g' | 'L' | 'ml' | 'piece' | 'pack'>('piece');
  const [category, setCategory] = useState('Fruit');
  const [purchaseDate, setPurchaseDate] = useState(new Date().toISOString().split('T')[0]);
  const [expiryDate, setExpiryDate] = useState(() => {
    const date = new Date();
    date.setDate(date.getDate() + 7); // Default: 7 days from now
    return date.toISOString().split('T')[0];
  });

  // States for barcode scanner
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    // Only run this when the scan dialog is opened
    if (!scanDialogOpen) {
        // Stop camera stream when dialog is closed
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        return;
    }

    const getCameraPermission = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        setHasCameraPermission(true);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
        setHasCameraPermission(false);
        toast({
          variant: 'destructive',
          title: 'Camera Access Denied',
          description: 'Please enable camera permissions in your browser settings to scan barcodes.',
        });
      }
    };

    getCameraPermission();
    
    // Cleanup function to stop media stream
    return () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
        }
    }
  }, [scanDialogOpen, toast]);

  const handleSimulateScan = async () => {
    setIsScanning(true);
    
    try {
      // Call ML API via Next.js API route to canonicalize the scanned item
      const response = await fetch('/api/canonicalize-item', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_text: "Tomatoes",
          method: "auto"
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setName(data.canonical_name || "Tomatoes");
        setCategory("Vegetable");
        setUnit("pack");
        setQuantity(1);
        
        toast({
            title: "Item Found!",
            description: `${data.canonical_name || "Tomatoes"} (pack) added. You can now adjust the details.`,
        });
      } else {
        // Fallback
        setName("Tomatoes");
        setCategory("Vegetable");
        setUnit("pack");
        setQuantity(1);
        
        toast({
            title: "Item Found!",
            description: "Tomatoes (pack) added. You can now adjust the details.",
        });
      }
    } catch (error) {
      console.error('Failed to canonicalize item:', error);
      // Fallback
      setName("Tomatoes");
      setCategory("Vegetable");
      setUnit("pack");
      setQuantity(1);
      
      toast({
          title: "Item Found!",
          description: "Tomatoes (pack) added. You can now adjust the details.",
      });
    }

    setIsScanning(false);
    setScanDialogOpen(false);
    setOpen(true);
  };

  const handleSubmit = () => {
    if (name && quantity > 0) {
      onAddItem({ name, quantity, unit, category, purchaseDate, expiryDate });
      setOpen(false);
      // Reset form
      setName('');
      setQuantity(1);
      setUnit('piece');
      setCategory('Fruit');
      setPurchaseDate(new Date().toISOString().split('T')[0]);
      const defaultExpiry = new Date();
      defaultExpiry.setDate(defaultExpiry.getDate() + 7);
      setExpiryDate(defaultExpiry.toISOString().split('T')[0]);
    }
  };
  
  const resetForm = () => {
    setName('');
    setQuantity(1);
    setUnit('piece');
    setCategory('Fruit');
    setPurchaseDate(new Date().toISOString().split('T')[0]);
    const defaultExpiry = new Date();
    defaultExpiry.setDate(defaultExpiry.getDate() + 7);
    setExpiryDate(defaultExpiry.toISOString().split('T')[0]);
  }

  return (
    <>
      <Dialog open={open} onOpenChange={(isOpen) => { setOpen(isOpen); if (!isOpen) resetForm(); }}>
        <DialogTrigger asChild>
          <Button>
            <PlusCircle className="mr-2 h-4 w-4" /> Add Item
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Add to Pantry</DialogTitle>
            <DialogDescription>
              Manually add an item or scan a barcode.
            </DialogDescription>
          </DialogHeader>

          <Button variant="outline" className="w-full" onClick={() => { setOpen(false); setScanDialogOpen(true); }}>
            <Camera className="mr-2 h-4 w-4" />
            Scan Barcode
          </Button>

          <div className="relative py-4">
              <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">
                      Or add manually
                  </span>
              </div>
          </div>
          
          <div className="grid gap-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Item Name
              </Label>
              <Input id="name" value={name} onChange={(e) => setName(e.target.value)} className="col-span-3" placeholder="e.g., Organic Bananas" />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="quantity" className="text-right">
                Quantity
              </Label>
              <Input 
                id="quantity" 
                type="number" 
                min="1" 
                value={quantity} 
                onChange={(e) => {
                  const val = parseInt(e.target.value) || 1;
                  setQuantity(val);
                }} 
                className="col-span-3" 
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="unit" className="text-right">
                Unit
              </Label>
              <Select onValueChange={(value: any) => setUnit(value)} value={unit}>
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select a unit" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="piece">Piece</SelectItem>
                  <SelectItem value="pack">Pack</SelectItem>
                  <SelectItem value="kg">kg</SelectItem>
                  <SelectItem value="g">g</SelectItem>
                  <SelectItem value="L">L</SelectItem>
                  <SelectItem value="ml">ml</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="category" className="text-right">
                Category
              </Label>
              <Select onValueChange={(value: any) => setCategory(value)} value={category}>
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Fruit">Fruit</SelectItem>
                  <SelectItem value="Vegetable">Vegetable</SelectItem>
                  <SelectItem value="Dairy">Dairy</SelectItem>
                  <SelectItem value="Bakery">Bakery</SelectItem>
                  <SelectItem value="Meat">Meat</SelectItem>
                  <SelectItem value="Pantry">Pantry Staple</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="purchase-date" className="text-right">
                Purchase Date
              </Label>
              <Input id="purchase-date" type="date" value={purchaseDate} onChange={(e) => setPurchaseDate(e.target.value)} className="col-span-3" />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="expiry-date" className="text-right">
                Expiry Date
              </Label>
              <Input 
                id="expiry-date" 
                type="date" 
                value={expiryDate} 
                onChange={(e) => setExpiryDate(e.target.value)} 
                className="col-span-3" 
                min={purchaseDate}
              />
            </div>
          </div>
          <DialogFooter>
            <Button onClick={handleSubmit}>Save Item</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* Barcode Scan Dialog */}
      <Dialog open={scanDialogOpen} onOpenChange={setScanDialogOpen}>
          <DialogContent className="sm:max-w-md">
              <DialogHeader>
                  <DialogTitle>Scan Barcode</DialogTitle>
                  <DialogDescription>
                      Position the item's barcode in front of the camera.
                  </DialogDescription>
              </DialogHeader>
              <div className="relative flex items-center justify-center rounded-md overflow-hidden aspect-video bg-muted border">
                  <video ref={videoRef} className="w-full h-full object-cover" autoPlay muted playsInline />
                  {isScanning && (
                      <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center text-white">
                          <ScanLine className="h-16 w-16 animate-pulse" />
                          <p className="mt-2">Scanning...</p>
                      </div>
                  )}
                  {hasCameraPermission === false && (
                       <Alert variant="destructive" className="m-4">
                          <Camera className="h-4 w-4" />
                          <AlertTitle>Camera Access Required</AlertTitle>
                          <AlertDescription>
                            Please allow camera access to scan barcodes.
                          </AlertDescription>
                      </Alert>
                  )}
                  <div className="absolute top-1/2 left-0 w-full h-0.5 bg-red-500/70 animate-pulse" />
              </div>
              <DialogFooter className="sm:justify-between gap-2">
                 <DialogClose asChild>
                    <Button type="button" variant="secondary">
                        Cancel
                    </Button>
                  </DialogClose>
                  <Button type="button" onClick={handleSimulateScan} disabled={isScanning || !hasCameraPermission}>
                      {isScanning ? "Scanning..." : "Simulate Scan"}
                  </Button>
              </DialogFooter>
          </DialogContent>
      </Dialog>
    </>
  );
}

    