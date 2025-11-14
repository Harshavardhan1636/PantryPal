'use client';
import { useUser } from '@/hooks/use-user';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { Header } from "@/components/dashboard/header";
import { Dashboard } from "@/components/dashboard/dashboard";

export default function Home() {
  const { user, loading, isProfileComplete } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
      return;
    }
    
    if (!loading && user && !isProfileComplete) {
      router.push('/profile');
      return;
    }
  }, [user, loading, isProfileComplete, router]);

  if (loading) {
    return (
      <div className="flex min-h-screen w-full flex-col items-center justify-center">
        <p>Loading...</p>
      </div>
    );
  }

  if (!user || !isProfileComplete) {
    return (
       <div className="flex min-h-screen w-full flex-col items-center justify-center">
        <p>Redirecting...</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen w-full flex-col">
      <Header />
      <main className="flex-1">
        <Dashboard />
      </main>
    </div>
  );
}
