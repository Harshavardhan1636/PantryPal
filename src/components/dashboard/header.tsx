import Image from "next/image";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { getPlaceholderImage } from "@/lib/placeholder-images";
import { useUser } from "@/hooks/use-user";
import { signOut } from "@/lib/local-auth";
import { Carrot, LogOut, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function Header() {
  const { user, refresh } = useUser();
  const userAvatar = getPlaceholderImage("user-avatar");
  const { toast } = useToast();
  const router = useRouter();

  const handleSignOut = async () => {
    try {
      signOut();
      refresh();
      toast({
        title: "Signed Out",
        description: "You have been successfully signed out.",
      });
      router.push('/login');
    } catch (error) {
      console.error("Sign out error", error);
      toast({
        title: "Error",
        description: "Failed to sign out.",
        variant: "destructive",
      });
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-all duration-300">
      <div className="container flex h-14 sm:h-16 items-center px-4 sm:px-6">
        <div className="mr-2 sm:mr-4 flex">
          <a href="/" className="mr-2 sm:mr-6 flex items-center space-x-2 group">
            <Carrot className="h-5 w-5 sm:h-6 sm:w-6 text-primary transition-transform group-hover:rotate-12" />
            <span className="font-bold inline-block font-headline text-base sm:text-lg">
              PantryPal
            </span>
            <Badge variant="outline" className="ml-1 sm:ml-2 text-[9px] sm:text-[10px] px-1.5 sm:px-2 py-0 border-primary/30 hidden xs:inline-flex">
              <Sparkles className="h-2.5 w-2.5 mr-1 animate-pulse" />
              AI Active
            </Badge>
          </a>
        </div>
        <div className="flex flex-1 items-center justify-end space-x-4">
          {user && (
             <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-10 w-10 rounded-full p-0 hover:ring-2 hover:ring-primary/20 transition-all">
                  <Avatar className="h-10 w-10">
                    {userAvatar && (
                      <AvatarImage src={userAvatar.imageUrl} alt="User avatar" data-ai-hint={userAvatar.imageHint} />
                    )}
                    <AvatarFallback className="bg-primary/10 font-semibold text-primary">
                      {user.name?.charAt(0).toUpperCase() || 'U'}
                    </AvatarFallback>
                  </Avatar>
                  {user.isGuest && (
                    <Badge variant="secondary" className="absolute -bottom-1 -right-1 h-4 px-1 text-[8px]">
                      Guest
                    </Badge>
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{user.name || "My Account"}</p>
                    <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => router.push('/profile')} className="cursor-pointer">
                  <Sparkles className="mr-2 h-4 w-4" />
                  <span>Profile Settings</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut} className="cursor-pointer text-destructive focus:text-destructive">
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Sign Out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>
    </header>
  );
}
