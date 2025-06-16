import React, { createContext, useContext, useEffect, useState, useRef, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { useSnackbar } from 'notistack';

import { useAuth } from '../hooks/useAuth';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface WebSocketContextValue {
  socket: Socket | null;
  connected: boolean;
  subscribe: (channel: string, callback: (data: any) => void) => () => void;
  emit: (event: string, data: any) => void;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

const WebSocketContext = createContext<WebSocketContextValue | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const { user, isAuthenticated } = useAuth();
  const { enqueueSnackbar } = useSnackbar();
  
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  
  const subscriptionsRef = useRef<Map<string, Set<(data: any) => void>>>(new Map());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  
  const maxReconnectAttempts = 5;
  const reconnectDelay = 1000; // Start with 1 second

  useEffect(() => {
    if (!isAuthenticated || !user) {
      if (socket) {
        socket.disconnect();
        setSocket(null);
        setConnected(false);
        setConnectionStatus('disconnected');
      }
      return;
    }

    // Initialize WebSocket connection
    const initializeSocket = () => {
      setConnectionStatus('connecting');
      
      const newSocket = io('/ws', {
        auth: {
          token: localStorage.getItem('access_token'),
          userId: user.userId,
          role: user.role,
        },
        transports: ['websocket', 'polling'],
        timeout: 10000,
        forceNew: true,
      });

      // Connection events
      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnected(true);
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        
        enqueueSnackbar('Real-time connection established', { 
          variant: 'success',
          autoHideDuration: 3000,
        });

        // Subscribe to user-specific events
        newSocket.emit('join_user_room', user.userId);
        
        // Subscribe to role-specific events
        newSocket.emit('join_role_room', user.role);
      });

      newSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setConnected(false);
        setConnectionStatus('disconnected');
        
        if (reason === 'io server disconnect') {
          // Server disconnected, don't reconnect automatically
          return;
        }

        // Attempt to reconnect
        attemptReconnect();
      });

      newSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnectionStatus('error');
        attemptReconnect();
      });

      // Business event handlers
      newSocket.on('claim_processed', (data) => {
        enqueueSnackbar(`Claim ${data.claimId} processed successfully`, { 
          variant: 'success' 
        });
        notifySubscribers('claim_processed', data);
      });

      newSocket.on('claim_failed', (data) => {
        enqueueSnackbar(`Claim ${data.claimId} failed: ${data.reason}`, { 
          variant: 'error' 
        });
        notifySubscribers('claim_failed', data);
      });

      newSocket.on('batch_completed', (data) => {
        enqueueSnackbar(
          `Batch ${data.batchId} completed: ${data.processedClaims}/${data.totalClaims} claims`, 
          { variant: 'info' }
        );
        notifySubscribers('batch_completed', data);
      });

      newSocket.on('system_alert', (data) => {
        enqueueSnackbar(data.message, { 
          variant: data.severity || 'warning',
          persist: data.severity === 'error',
        });
        notifySubscribers('system_alert', data);
      });

      newSocket.on('throughput_update', (data) => {
        notifySubscribers('throughput_update', data);
      });

      newSocket.on('failed_claims_update', (data) => {
        notifySubscribers('failed_claims', data);
      });

      newSocket.on('user_notification', (data) => {
        enqueueSnackbar(data.message, { 
          variant: data.type || 'info',
          autoHideDuration: data.autoHide !== false ? 5000 : null,
        });
      });

      // Generic message handler
      newSocket.onAny((eventName, data) => {
        notifySubscribers(eventName, data);
      });

      setSocket(newSocket);
    };

    const attemptReconnect = () => {
      if (reconnectAttempts.current >= maxReconnectAttempts) {
        enqueueSnackbar('Failed to establish real-time connection', { 
          variant: 'error',
          persist: true,
        });
        return;
      }

      const delay = reconnectDelay * Math.pow(2, reconnectAttempts.current);
      reconnectAttempts.current++;

      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})`);
        initializeSocket();
      }, delay);
    };

    const notifySubscribers = (channel: string, data: any) => {
      const subscribers = subscriptionsRef.current.get(channel);
      if (subscribers) {
        subscribers.forEach(callback => {
          try {
            callback(data);
          } catch (error) {
            console.error('Error in WebSocket subscriber callback:', error);
          }
        });
      }
    };

    initializeSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (socket) {
        socket.disconnect();
      }
    };
  }, [isAuthenticated, user, enqueueSnackbar]);

  const subscribe = (channel: string, callback: (data: any) => void): (() => void) => {
    if (!subscriptionsRef.current.has(channel)) {
      subscriptionsRef.current.set(channel, new Set());
    }
    
    subscriptionsRef.current.get(channel)!.add(callback);

    // Return unsubscribe function
    return () => {
      const subscribers = subscriptionsRef.current.get(channel);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          subscriptionsRef.current.delete(channel);
        }
      }
    };
  };

  const emit = (event: string, data: any) => {
    if (socket && connected) {
      socket.emit(event, data);
    } else {
      console.warn('Cannot emit event: WebSocket not connected');
    }
  };

  const contextValue: WebSocketContextValue = {
    socket,
    connected,
    subscribe,
    emit,
    connectionStatus,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (channel?: string, callback?: (data: any) => void) => {
  const context = useContext(WebSocketContext);
  
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }

  useEffect(() => {
    if (channel && callback) {
      return context.subscribe(channel, callback);
    }
  }, [channel, callback, context]);

  return context;
};

export default WebSocketContext;