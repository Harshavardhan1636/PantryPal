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
import { useEffect } from 'react';
import { updateHousehold, getCurrentHousehold } from '@/lib/local-auth';

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
        description: 'Your household profile has been updated.',
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
  
  if (loading) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>;
  }
  
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Household Profile</CardTitle>
          <CardDescription>
            {isProfileComplete ? 'Edit your household details below.' : "Welcome! Let's finish setting up your household profile."}
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="householdSize">Household Size</Label>
              <Controller
                name="householdSize"
                control={control}
                render={({ field }) => (
                  <Input id="householdSize" type="number" {...field} />
                )}
              />
              {errors.householdSize && <p className="text-sm text-destructive">{errors.householdSize.message}</p>}
            </div>
            <div className="space-y-2">
              <Label htmlFor="dietaryPreferences">Dietary Preferences (comma-separated)</Label>
              <Controller
                name="dietaryPreferences"
                control={control}
                render={({ field }) => (
                  <Input id="dietaryPreferences" placeholder="e.g., Vegetarian, Gluten-Free" {...field} />
                )}
              />
              {errors.dietaryPreferences && <p className="text-sm text-destructive">{errors.dietaryPreferences.message}</p>}
            </div>
          </CardContent>
          <CardFooter className="flex justify-end">
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? 'Saving...' : 'Save and Continue'}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
