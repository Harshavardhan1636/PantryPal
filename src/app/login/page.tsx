'use client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { useUser } from '@/hooks/use-user';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Carrot, User, Mail, Sparkles, TrendingDown, Brain, ChefHat, Lock, Shield } from 'lucide-react';
import { signInWithEmail, signInAsGuest, updateHousehold } from '@/lib/local-auth';
import { STORAGE_KEYS, setToStorage } from '@/lib/local-storage';
import { mockPantryItems } from '@/lib/mock-data';
import { Badge } from '@/components/ui/badge';

export default function LoginPage() {
  const { toast } = useToast();
  const { user, loading, refresh } = useUser();
  const router = useRouter();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!loading && user) {
      router.push('/');
    }
  }, [user, loading, router]);

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) {
      toast({
        title: 'Missing Information',
        description: 'Please enter your name and email.',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    try {
      signInWithEmail(email, name);
      updateHousehold({
        name: `${name}'s Household`,
        membersCount: 1,
      });
      // Start with empty pantry for real users
      setToStorage(STORAGE_KEYS.PANTRY_ITEMS, []);
      toast({
        title: 'Welcome!',
        description: "Let's get your household set up.",
      });
      refresh();
      router.push('/profile');
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to sign in.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGuestSignIn = async () => {
    setIsLoading(true);
    try {
      signInAsGuest();
      updateHousehold({
        name: 'Guest Household',
        membersCount: 2,
        dietaryPreferences: ['Vegetarian'],
      });
      const pantryItems = mockPantryItems.map((item, index) => ({
        ...item,
        id: `mock-${index}`,
      }));
      setToStorage(STORAGE_KEYS.PANTRY_ITEMS, pantryItems);
      toast({
        title: 'Guest Mode',
        description: 'Welcome! Explore with demo data.',
      });
      refresh();
      router.push('/');
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to sign in as guest.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center space-y-4 animate-pulse">
          <Carrot className="h-12 w-12 text-primary mx-auto" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-primary/10 via-background to-accent/10 p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-md space-y-6 sm:space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
        {/* Logo & Branding */}
        <div className="text-center space-y-3 sm:space-y-4">
          <div className="mx-auto flex h-20 w-20 sm:h-24 sm:w-24 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/60 shadow-lg animate-in zoom-in duration-500">
            <Carrot className="h-12 w-12 text-white" />
          </div>
          <div className="space-y-2">
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              PantryPal
            </h1>
            <p className="text-sm sm:text-base lg:text-lg text-muted-foreground">
              Reduce food waste with AI-powered insights
            </p>
          </div>
          
          {/* Feature badges */}
          <div className="flex flex-wrap justify-center gap-2 pt-2">
            <Badge variant="secondary" className="gap-1">
              <Brain className="h-3 w-3" />
              AI Predictions
            </Badge>
            <Badge variant="secondary" className="gap-1">
              <ChefHat className="h-3 w-3" />
              Smart Recipes
            </Badge>
            <Badge variant="secondary" className="gap-1">
              <TrendingDown className="h-3 w-3" />
              Waste Reduction
            </Badge>
          </div>
        </div>

        {/* Login Card */}
        <Card className="shadow-2xl border-2 transition-all duration-300 hover:shadow-primary/20">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl">Welcome Back</CardTitle>
            <CardDescription>Sign in to start managing your pantry intelligently</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <form onSubmit={handleEmailSignIn} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name" className="text-sm font-medium">Name</Label>
                <div className="relative">
                  <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="name"
                    placeholder="John Doe"
                    className="pl-10 transition-all focus:ring-2 focus:ring-primary/50"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    disabled={isLoading}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="john@example.com"
                    className="pl-10 transition-all focus:ring-2 focus:ring-primary/50"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={isLoading}
                  />
                </div>
              </div>
              <Button 
                type="submit" 
                className="w-full transition-all hover:scale-[1.02]" 
                disabled={isLoading}
                size="lg"
              >
                {isLoading ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-background border-t-transparent mr-2" />
                    Signing In...
                  </>
                ) : (
                  'Continue'
                )}
              </Button>
            </form>
            
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground font-medium">Or</span>
              </div>
            </div>
            
            <div className="space-y-2">
              <Button
                variant="outline"
                className="w-full transition-all hover:scale-[1.02] border-2"
                onClick={handleGuestSignIn}
                disabled={isLoading}
                size="lg"
              >
                <Sparkles className="mr-2 h-4 w-4" />
                Try Demo with Sample Data
              </Button>
              <p className="text-xs text-center text-muted-foreground">
                Explore all features with pre-loaded pantry items
              </p>
            </div>
          </CardContent>
        </Card>
        
        {/* Footer */}
        <div className="space-y-2">
          <p className="text-center text-sm text-muted-foreground flex items-center justify-center gap-2">
            <Shield className="h-4 w-4" />
            No passwords required. Your data is stored locally in your browser.
          </p>
          <p className="text-center text-xs text-muted-foreground">
            Privacy-first • No tracking • No external servers
          </p>
        </div>
      </div>
    </div>
  );
}
