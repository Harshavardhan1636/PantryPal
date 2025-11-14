'use client';

import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { useRouter } from 'next/navigation';
import { useUser } from '@/hooks/use-user';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { useEffect, useState } from 'react';
import { updateHousehold, getCurrentHousehold, signOut } from '@/lib/local-auth';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Users, Home, Utensils, LogOut, Save, ArrowLeft, TrendingUp, Package } from 'lucide-react';
import { getFromStorage, STORAGE_KEYS } from '@/lib/local-storage';
import type { PantryItem } from '@/lib/types';

const profileSchema = z.object({
  householdSize: z.coerce.number().min(1, 'Household size must be at least 1.'),
  dietaryPreferences: z.string().optional(),
});

type ProfileFormValues = z.infer<typeof profileSchema>;

export default function ProfilePage() {
  const { user, loading, household, isProfileComplete, refresh } = useUser();
  const router = useRouter();
  const { toast } = useToast();

  const { control, handleSubmit, formState: { errors, isSubmitting }, setValue } = useForm<ProfileFormValues>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      householdSize: 1,
      dietaryPreferences: '',
    },
  });

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  useEffect(() => {
    if (household) {
      setValue('householdSize', household.membersCount || 1);
      setValue('dietaryPreferences', household.dietaryPreferences?.join(', ') || '');
    }
  }, [household, setValue]);

  const [pantryStats, setPantryStats] = useState({ total: 0, highRisk: 0, categories: 0 });

  useEffect(() => {
    const items = getFromStorage<PantryItem[]>(STORAGE_KEYS.PANTRY_ITEMS) || [];
    const highRisk = items.filter(item => item.riskClass === 'High').length;
    const categories = new Set(items.map(item => item.category)).size;
    setPantryStats({ total: items.length, highRisk, categories });
  }, []);

  const onSubmit = async (data: ProfileFormValues) => {
    if (!user) {
      toast({ title: 'Error', description: 'You must be logged in.', variant: 'destructive' });
      return;
    }

    try {
      updateHousehold({
        membersCount: data.householdSize,
        dietaryPreferences: data.dietaryPreferences?.split(',').map(s => s.trim()).filter(Boolean) || [],
      });
        
      toast({
        title: 'Profile Saved',
        description: 'Your household profile has been updated successfully!',
        duration: 3000,
      });

      refresh();
      router.push('/');
    } catch (error) {
      console.error('Failed to save profile:', error);
      toast({
        title: 'Error',
        description: 'Failed to save your profile. Please try again.',
        variant: 'destructive',
      });
    }
  };

  const handleSignOut = () => {
    signOut();
    toast({
      title: 'Signed Out',
      description: 'You have been signed out successfully.',
    });
    router.push('/login');
  };
  
  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="animate-pulse space-y-4">
          <Package className="h-12 w-12 text-primary mx-auto" />
          <p className="text-muted-foreground">Loading profile...</p>
        </div>
      </div>
    );
  }
  
  const userInitials = user?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'U';
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-primary/5 to-background p-4 sm:p-6 lg:p-8">
      <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 sm:gap-0">
          <Button variant="ghost" onClick={() => router.push('/')} className="gap-2 w-full sm:w-auto">
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">Back to Dashboard</span>
            <span className="sm:hidden">Back</span>
          </Button>
          <Button variant="outline" onClick={handleSignOut} className="gap-2 text-destructive hover:text-destructive w-full sm:w-auto">
            <LogOut className="h-4 w-4" />
            Sign Out
          </Button>
        </div>

        {/* User Info Card */}
        <Card className="border-2">
          <CardHeader>
            <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
              <Avatar className="h-16 w-16 sm:h-20 sm:w-20 border-2 border-primary">
                <AvatarFallback className="bg-primary/10 text-2xl font-bold text-primary">
                  {userInitials}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <CardTitle className="text-2xl">{user?.name || 'User'}</CardTitle>
                <CardDescription className="flex items-center gap-2 mt-1">
                  {user?.email || 'guest@pantrypal.local'}
                  {user?.isGuest && (
                    <Badge variant="secondary" className="text-xs">Guest</Badge>
                  )}
                </CardDescription>
              </div>
            </div>
          </CardHeader>
        </Card>

        {/* Statistics */}
        <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-3">
          <Card className="transition-all hover:shadow-lg">
            <CardHeader className="pb-3">
              <CardDescription className="flex items-center gap-2">
                <Package className="h-4 w-4" />
                Total Items
              </CardDescription>
              <CardTitle className="text-3xl">{pantryStats.total}</CardTitle>
            </CardHeader>
          </Card>
          <Card className="transition-all hover:shadow-lg">
            <CardHeader className="pb-3">
              <CardDescription className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                High Risk Items
              </CardDescription>
              <CardTitle className="text-3xl text-destructive">{pantryStats.highRisk}</CardTitle>
            </CardHeader>
          </Card>
          <Card className="transition-all hover:shadow-lg">
            <CardHeader className="pb-3">
              <CardDescription className="flex items-center gap-2">
                <Utensils className="h-4 w-4" />
                Categories
              </CardDescription>
              <CardTitle className="text-3xl">{pantryStats.categories}</CardTitle>
            </CardHeader>
          </Card>
        </div>

        {/* Profile Settings */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Home className="h-5 w-5 text-primary" />
              <CardTitle>Household Settings</CardTitle>
            </div>
            <CardDescription>
              {isProfileComplete ? 'Update your household details below.' : "Let's finish setting up your household profile."}
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit(onSubmit)}>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="householdSize" className="flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  Household Size
                </Label>
                <Controller
                  name="householdSize"
                  control={control}
                  render={({ field }) => (
                    <Input 
                      id="householdSize" 
                      type="number" 
                      min="1"
                      max="20"
                      {...field}
                      className="max-w-xs"
                    />
                  )}
                />
                {errors.householdSize && <p className="text-sm text-destructive">{errors.householdSize.message}</p>}
                <p className="text-xs text-muted-foreground">Number of people in your household</p>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <Label htmlFor="dietaryPreferences" className="flex items-center gap-2">
                  <Utensils className="h-4 w-4" />
                  Dietary Preferences
                </Label>
                <Controller
                  name="dietaryPreferences"
                  control={control}
                  render={({ field }) => (
                    <Input 
                      id="dietaryPreferences" 
                      placeholder="e.g., Vegetarian, Gluten-Free, Vegan" 
                      {...field}
                    />
                  )}
                />
                {errors.dietaryPreferences && <p className="text-sm text-destructive">{errors.dietaryPreferences.message}</p>}
                <p className="text-xs text-muted-foreground">Separate multiple preferences with commas</p>
                {household?.dietaryPreferences && household.dietaryPreferences.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-2">
                    {household.dietaryPreferences.map((pref, idx) => (
                      <Badge key={idx} variant="secondary">{pref}</Badge>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button type="button" variant="outline" onClick={() => router.push('/')}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting} className="gap-2">
                {isSubmitting ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-background border-t-transparent" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save Changes
                  </>
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>

        {/* Footer Info */}
        <Card className="bg-muted/50">
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground text-center">
              Your data is stored locally in your browser. No external servers are used.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
