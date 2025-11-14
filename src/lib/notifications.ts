/**
 * Browser notification system for PantryPal
 */

export type NotificationType = 'high-risk' | 'expiring-soon' | 'recipe-suggestion' | 'waste-saved';

export type NotificationData = {
  id: string;
  type: NotificationType;
  title: string;
  body: string;
  timestamp: string;
  read: boolean;
};

/**
 * Request notification permission from user
 */
export async function requestNotificationPermission(): Promise<boolean> {
  if (!('Notification' in window)) {
    console.warn('This browser does not support notifications');
    return false;
  }

  if (Notification.permission === 'granted') {
    return true;
  }

  if (Notification.permission !== 'denied') {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }

  return false;
}

/**
 * Show a browser notification
 */
export function showNotification(title: string, options?: NotificationOptions): void {
  if (!('Notification' in window)) {
    console.warn('This browser does not support notifications');
    return;
  }

  if (Notification.permission === 'granted') {
    new Notification(title, {
      icon: '/icon-192.png',
      badge: '/icon-192.png',
      ...options,
    });
  }
}

/**
 * Schedule a notification for at-risk items
 */
export function notifyAboutAtRiskItems(itemNames: string[]): void {
  if (itemNames.length === 0) return;

  const title = '🚨 Food Waste Alert!';
  const body = `${itemNames.length} items are at high risk: ${itemNames.slice(0, 3).join(', ')}${itemNames.length > 3 ? '...' : ''}`;

  showNotification(title, {
    body,
    tag: 'high-risk-items',
    requireInteraction: true,
  });
}

/**
 * Notify about expiring items
 */
export function notifyAboutExpiringItems(itemNames: string[]): void {
  if (itemNames.length === 0) return;

  const title = '⏰ Items Expiring Soon';
  const body = `${itemNames.length} items expiring in the next 3 days: ${itemNames.slice(0, 3).join(', ')}`;

  showNotification(title, {
    body,
    tag: 'expiring-soon',
  });
}

/**
 * Check and notify about at-risk items (can be called periodically)
 */
export function checkAndNotify(pantryItems: Array<{ name: string; riskClass?: string; expiryDate?: string }>): void {
  // High risk items
  const highRiskItems = pantryItems
    .filter(item => item.riskClass === 'High')
    .map(item => item.name);

  if (highRiskItems.length > 0) {
    notifyAboutAtRiskItems(highRiskItems);
  }

  // Items expiring soon (within 3 days)
  const threeDaysFromNow = new Date();
  threeDaysFromNow.setDate(threeDaysFromNow.getDate() + 3);

  const expiringItems = pantryItems
    .filter(item => {
      if (!item.expiryDate) return false;
      const expiryDate = new Date(item.expiryDate);
      return expiryDate <= threeDaysFromNow;
    })
    .map(item => item.name);

  if (expiringItems.length > 0) {
    notifyAboutExpiringItems(expiringItems);
  }
}
