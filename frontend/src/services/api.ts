import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

// Types
export interface User {
  userId: string;
  username: string;
  role: string;
  email?: string;
  permissions: string[];
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user_info: User;
}

export interface FailedClaim {
  id: number;
  claimReference: string;
  facilityId: string;
  failureCategory: string;
  failureReason: string;
  failureDetails: any;
  chargeAmount: number;
  expectedReimbursement?: number;
  resolutionStatus: string;
  assignedTo?: string;
  resolutionNotes?: string;
  failedAt: string;
  createdAt: string;
  updatedAt: string;
  canReprocess: boolean;
  reprocessCount: number;
}

export interface ClaimFilters {
  facilityId?: string;
  failureCategory?: string;
  resolutionStatus?: string;
  dateRange: {
    startDate: Date | null;
    endDate: Date | null;
  };
  priority?: string;
  assignedTo?: string;
  limit?: number;
  offset?: number;
}

export interface FailedClaimsResponse {
  failedClaims: FailedClaim[];
  totalCount: number;
  limit: number;
  offset: number;
}

export interface Claim {
  claimId: string;
  facilityId: string;
  patientAccountNumber: string;
  patientFirstName?: string;
  patientLastName?: string;
  patientDateOfBirth?: string;
  totalCharges: number;
  processingStatus: string;
  createdAt: string;
  lineItems?: ClaimLineItem[];
}

export interface ClaimLineItem {
  lineNumber: number;
  procedureCode: string;
  procedureDescription?: string;
  units: number;
  chargeAmount: number;
  serviceDate: string;
}

export interface BatchSubmissionRequest {
  facilityId: string;
  claims: any[];
  priority?: string;
  submittedBy: string;
}

export interface BatchSubmissionResponse {
  batchId: string;
  status: string;
  claimsCount: number;
  estimatedCompletion: string;
}

export interface BatchStatus {
  batchId: string;
  status: string;
  totalClaims: number;
  processedClaims: number;
  failedClaims: number;
  processingTime?: number;
  throughput?: number;
  startedAt?: string;
  completedAt?: string;
}

export interface ResolutionRequest {
  claimId: string;
  resolutionType: 'manual_fix' | 'reprocess' | 'reject' | 'escalate';
  resolutionNotes: string;
  correctedData?: any;
  businessJustification?: string;
}

export interface DashboardMetrics {
  totalClaimsToday: number;
  successfulClaimsToday: number;
  failedClaimsToday: number;
  avgProcessingTime: number;
  currentThroughput: number;
  targetThroughput: number;
  systemHealth: 'healthy' | 'degraded' | 'unhealthy';
  recentBatches: BatchStatus[];
  failuresByCategory: { [key: string]: number };
}

export interface AnalyticsData {
  processingTrends: any[];
  facilityPerformance: any[];
  errorAnalysis: any[];
  revenueMetrics: any[];
  throughputMetrics: any[];
}

// API Client Configuration
class ApiClient {
  private client: AxiosInstance;
  private authToken: string | null = null;

  constructor(baseURL: string = '/api/v1') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        
        // Add request ID for tracking
        config.headers['X-Request-ID'] = this.generateRequestId();
        
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          
          try {
            await this.refreshToken();
            return this.client(originalRequest);
          } catch (refreshError) {
            this.logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );

    // Load token from localStorage on initialization
    const savedToken = localStorage.getItem('access_token');
    if (savedToken) {
      this.setAuthToken(savedToken);
    }
  }

  private generateRequestId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  setAuthToken(token: string): void {
    this.authToken = token;
    localStorage.setItem('access_token', token);
  }

  clearAuthToken(): void {
    this.authToken = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  async refreshToken(): Promise<void> {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await axios.post('/auth/refresh', {
      refresh_token: refreshToken,
    });

    this.setAuthToken(response.data.access_token);
  }

  logout(): void {
    this.clearAuthToken();
  }

  // Authentication endpoints
  async login(credentials: LoginRequest): Promise<LoginResponse> {
    const response = await axios.post('/auth/login', credentials);
    const loginData = response.data;
    
    this.setAuthToken(loginData.access_token);
    localStorage.setItem('refresh_token', loginData.refresh_token);
    
    return loginData;
  }

  async logoutUser(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      this.logout();
    }
  }

  // Health endpoints
  async getHealth(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async getDetailedHealth(): Promise<any> {
    const response = await this.client.get('/health/detailed');
    return response.data;
  }

  // Claims endpoints
  async getClaim(claimId: string, businessJustification?: string): Promise<Claim> {
    const config: AxiosRequestConfig = {};
    if (businessJustification) {
      config.params = { business_justification: businessJustification };
    }
    
    const response = await this.client.get(`/claims/${claimId}`, config);
    return response.data;
  }

  async submitBatch(batchData: BatchSubmissionRequest): Promise<BatchSubmissionResponse> {
    const response = await this.client.post('/claims/batch', batchData);
    return response.data;
  }

  async getBatchStatus(batchId: string): Promise<BatchStatus> {
    const response = await this.client.get(`/batches/${batchId}/status`);
    return response.data;
  }

  // Failed claims endpoints
  async getFailedClaims(filters: ClaimFilters): Promise<FailedClaimsResponse> {
    const params: any = {};
    
    if (filters.facilityId) params.facility_id = filters.facilityId;
    if (filters.failureCategory) params.failure_category = filters.failureCategory;
    if (filters.resolutionStatus) params.resolution_status = filters.resolutionStatus;
    if (filters.priority) params.priority = filters.priority;
    if (filters.assignedTo) params.assigned_to = filters.assignedTo;
    if (filters.limit) params.limit = filters.limit;
    if (filters.offset) params.offset = filters.offset;
    
    if (filters.dateRange.startDate) {
      params.start_date = filters.dateRange.startDate.toISOString();
    }
    if (filters.dateRange.endDate) {
      params.end_date = filters.dateRange.endDate.toISOString();
    }

    const response = await this.client.get('/failed-claims', { params });
    return response.data;
  }

  async resolveClaim(claimId: string, resolutionData: Omit<ResolutionRequest, 'claimId'>): Promise<void> {
    await this.client.post(`/failed-claims/${claimId}/resolve`, resolutionData);
  }

  async assignClaims(claimIds: string[], assignedTo: string): Promise<void> {
    await this.client.post('/failed-claims/assign', {
      claim_ids: claimIds,
      assigned_to: assignedTo,
    });
  }

  async exportClaims(claimIds: string[]): Promise<Blob> {
    const response = await this.client.post('/failed-claims/export', 
      { claim_ids: claimIds },
      { responseType: 'blob' }
    );
    return response.data;
  }

  // Dashboard and analytics endpoints
  async getDashboardMetrics(): Promise<DashboardMetrics> {
    const response = await this.client.get('/dashboard/metrics');
    return response.data;
  }

  async getAnalyticsData(dateRange: { start: Date; end: Date }): Promise<AnalyticsData> {
    const response = await this.client.get('/analytics', {
      params: {
        start_date: dateRange.start.toISOString(),
        end_date: dateRange.end.toISOString(),
      },
    });
    return response.data;
  }

  // Reports endpoints
  async generateReport(reportType: string, filters: any): Promise<Blob> {
    const response = await this.client.post(`/reports/${reportType}`, filters, {
      responseType: 'blob',
    });
    return response.data;
  }
}

// Create API client instance
const apiClient = new ApiClient();

// Export specific API modules
export const authApi = {
  login: (credentials: LoginRequest) => apiClient.login(credentials),
  logout: () => apiClient.logoutUser(),
  setToken: (token: string) => apiClient.setAuthToken(token),
  clearToken: () => apiClient.clearAuthToken(),
};

export const healthApi = {
  getHealth: () => apiClient.getHealth(),
  getDetailedHealth: () => apiClient.getDetailedHealth(),
};

export const claimsApi = {
  getClaim: (claimId: string, businessJustification?: string) => 
    apiClient.getClaim(claimId, businessJustification),
  submitBatch: (batchData: BatchSubmissionRequest) => apiClient.submitBatch(batchData),
  getBatchStatus: (batchId: string) => apiClient.getBatchStatus(batchId),
};

export const failedClaimsApi = {
  getFailedClaims: (filters: ClaimFilters) => apiClient.getFailedClaims(filters),
  resolveClaim: (claimId: string, resolutionData: Omit<ResolutionRequest, 'claimId'>) =>
    apiClient.resolveClaim(claimId, resolutionData),
  assignClaims: (claimIds: string[], assignedTo: string) =>
    apiClient.assignClaims(claimIds, assignedTo),
  exportClaims: (claimIds: string[]) => apiClient.exportClaims(claimIds),
};

export const dashboardApi = {
  getMetrics: () => apiClient.getDashboardMetrics(),
};

export const analyticsApi = {
  getData: (dateRange: { start: Date; end: Date }) => apiClient.getAnalyticsData(dateRange),
};

export const reportsApi = {
  generate: (reportType: string, filters: any) => apiClient.generateReport(reportType, filters),
};

export default apiClient;