/**
 * Frontend logging utility for error tracking and debugging.
 * Sends logs to backend for centralized storage.
 */

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error'
}

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: Record<string, any>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
  userAgent?: string;
  url?: string;
  userId?: string;
  sessionId?: string;
}

class FrontendLogger {
  private readonly apiEndpoint = '/api/v1/logs/frontend';
  private logQueue: LogEntry[] = [];
  private flushInterval: number = 5000; // 5 seconds
  private maxQueueSize: number = 100;
  private sessionId: string;
  private userId?: string;

  constructor() {
    this.sessionId = this.generateSessionId();
    this.startFlushTimer();
    this.setupErrorHandlers();
  }

  private generateSessionId(): string {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  }

  private startFlushTimer(): void {
    setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }

  private setupErrorHandlers(): void {
    // Global error handler
    window.addEventListener('error', (event) => {
      this.error('Uncaught error', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error ? {
          name: event.error.name,
          message: event.error.message,
          stack: event.error.stack
        } : undefined
      });
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      this.error('Unhandled promise rejection', {
        reason: event.reason,
        promise: event.promise.toString()
      });
    });
  }

  setUserId(userId: string): void {
    this.userId = userId;
  }

  private createLogEntry(level: LogLevel, message: string, context?: Record<string, any>, error?: Error): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
      userAgent: navigator.userAgent,
      url: window.location.href,
      userId: this.userId,
      sessionId: this.sessionId
    };

    if (error) {
      entry.error = {
        name: error.name,
        message: error.message,
        stack: error.stack
      };
    }

    return entry;
  }

  private addToQueue(entry: LogEntry): void {
    this.logQueue.push(entry);

    // Log to console for development
    if (process.env.NODE_ENV === 'development') {
      const logMethod = entry.level === 'error' ? console.error :
                       entry.level === 'warn' ? console.warn :
                       entry.level === 'info' ? console.info : console.debug;
      
      logMethod(`[${entry.level.toUpperCase()}] ${entry.message}`, entry.context || '', entry.error || '');
    }

    // Flush immediately if queue is getting large
    if (this.logQueue.length >= this.maxQueueSize) {
      this.flush();
    }

    // Flush immediately for errors
    if (entry.level === LogLevel.ERROR) {
      this.flush();
    }
  }

  debug(message: string, context?: Record<string, any>): void {
    const entry = this.createLogEntry(LogLevel.DEBUG, message, context);
    this.addToQueue(entry);
  }

  info(message: string, context?: Record<string, any>): void {
    const entry = this.createLogEntry(LogLevel.INFO, message, context);
    this.addToQueue(entry);
  }

  warn(message: string, context?: Record<string, any>): void {
    const entry = this.createLogEntry(LogLevel.WARN, message, context);
    this.addToQueue(entry);
  }

  error(message: string, context?: Record<string, any>, error?: Error): void {
    const entry = this.createLogEntry(LogLevel.ERROR, message, context, error);
    this.addToQueue(entry);
  }

  async flush(): Promise<void> {
    if (this.logQueue.length === 0) {
      return;
    }

    const logsToSend = [...this.logQueue];
    this.logQueue = [];

    try {
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ logs: logsToSend })
      });

      if (!response.ok) {
        // If send fails, add logs back to queue (but limit to prevent infinite growth)
        if (this.logQueue.length < this.maxQueueSize / 2) {
          this.logQueue.unshift(...logsToSend.slice(-10)); // Keep only last 10
        }
        console.warn('Failed to send logs to backend:', response.statusText);
      }
    } catch (error) {
      // Network error - add logs back to queue
      if (this.logQueue.length < this.maxQueueSize / 2) {
        this.logQueue.unshift(...logsToSend.slice(-10)); // Keep only last 10
      }
      console.warn('Failed to send logs to backend:', error);
    }
  }

  // Utility methods for common logging scenarios
  logApiError(endpoint: string, error: Error, requestData?: any): void {
    this.error('API request failed', {
      endpoint,
      requestData,
      errorType: 'API_ERROR'
    }, error);
  }

  logUserAction(action: string, details?: Record<string, any>): void {
    this.info('User action', {
      action,
      ...details,
      type: 'USER_ACTION'
    });
  }

  logPageView(page: string, additionalData?: Record<string, any>): void {
    this.info('Page view', {
      page,
      ...additionalData,
      type: 'PAGE_VIEW'
    });
  }

  logPerformance(metric: string, value: number, context?: Record<string, any>): void {
    this.info('Performance metric', {
      metric,
      value,
      ...context,
      type: 'PERFORMANCE'
    });
  }

  // Method to manually trigger log send (useful for page unload)
  async sendImmediately(): Promise<void> {
    await this.flush();
  }
}

// Create singleton instance
export const logger = new FrontendLogger();

// Setup page unload handler to send remaining logs
window.addEventListener('beforeunload', () => {
  // Use sendBeacon for more reliable delivery during page unload
  if (logger.logQueue.length > 0 && navigator.sendBeacon) {
    const payload = JSON.stringify({ logs: logger.logQueue });
    navigator.sendBeacon('/api/v1/logs/frontend', payload);
  }
});

// Export convenience functions
export const logError = (message: string, error?: Error, context?: Record<string, any>) => {
  logger.error(message, context, error);
};

export const logInfo = (message: string, context?: Record<string, any>) => {
  logger.info(message, context);
};

export const logWarning = (message: string, context?: Record<string, any>) => {
  logger.warn(message, context);
};

export const logUserAction = (action: string, details?: Record<string, any>) => {
  logger.logUserAction(action, details);
};

export const logApiError = (endpoint: string, error: Error, requestData?: any) => {
  logger.logApiError(endpoint, error, requestData);
};

export default logger;